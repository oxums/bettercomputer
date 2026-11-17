"""Stage 0 smoke test. Answers, on whatever machine it runs on:

1. Does the base model load and train with our added parts?
2. Which number format (fp32 / fp16 / bf16) works and how fast is each?
3. Can the model actually learn to point? (synthetic task, before and after training)
4. Do checkpoints save, reload and resume correctly?
5. How long does one decision take?

Run:  python -m bettercomputer.smoke
Prints a JSON report between ===REPORT=== markers and writes it to --out.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch

from .data import Collator, synthetic_batches
from .model import SelectorConfig, SelectorModel
from .train import AMP_DTYPES, autocast, evaluate, to_device, train


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def format_check(model: SelectorModel, batch: dict, device: torch.device, amp: str, reps: int) -> dict:
    """A few forward+backward passes in one number format: is everything finite, how fast?"""
    params = [p for p in model.parameters() if p.requires_grad]
    tokens = int(batch["intent_mask"].sum() + batch["elem_mask"].sum()) + batch["target"].shape[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, loss_value, grads_finite = [], float("nan"), True
    try:
        for _ in range(reps + 1):  # first pass is warm-up
            sync(device)
            t = time.time()
            with autocast(device, amp):
                out = model(batch)
            loss = model.loss(out, batch)
            model.zero_grad(set_to_none=True)
            loss.backward()
            sync(device)
            times.append(time.time() - t)
            loss_value = loss.item()
            print(f"  {amp} pass {len(times)}: {times[-1]:.2f}s loss {loss_value:.3f}", flush=True)
            grads_finite = grads_finite and all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    except Exception as exc:  # a format can simply be unsupported on a device
        return {"ok": False, "error": repr(exc)[:300]}
    finally:
        model.zero_grad(set_to_none=True)
    step_s = statistics.median(times[1:])
    result = {
        "ok": bool(torch.isfinite(torch.tensor(loss_value)).item() and grads_finite),
        "loss": round(loss_value, 4),
        "grads_finite": grads_finite,
        "s_per_step": round(step_s, 4),
        "tokens_per_s": round(tokens / step_s),
    }
    if device.type == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return result


@torch.no_grad()
def decision_latency(model: SelectorModel, collator: Collator, device: torch.device, amp: str, n_elements: int, reps: int) -> dict:
    """Time for one decision (batch of 1) with a fixed number of elements on screen."""
    batch = to_device(next(synthetic_batches(collator, 1, seed=7, min_elements=n_elements, max_elements=n_elements, p_none=0.0)), device)
    model.eval()
    times = []
    for _ in range(reps + 3):
        sync(device)
        t = time.time()
        with autocast(device, amp):
            model(batch)
        sync(device)
        times.append(time.time() - t)
    model.train()
    return {"elements": int(batch["elem_mask"].sum()), "median_ms": round(1000 * statistics.median(times[3:]), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500, help="training steps for the learning test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--amp", default="auto", choices=["auto", *AMP_DTYPES])
    ap.add_argument("--formats", default="fp32,fp16", help="formats to check on a GPU (bf16 is slow on a T4; each one compiles kernels on first use)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-elements", type=int, default=40)
    ap.add_argument("--text-mode", default="tokens", choices=["tokens", "pooled"])
    ap.add_argument("--out", default="smoke_report.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else f"cpu ({platform.processor()})",
        "torch": torch.__version__,
    }
    import transformers

    report["transformers"] = transformers.__version__
    try:
        import fla  # noqa: F401  (flash-linear-attention: optimized kernel for the recurrent layers)

        report["recurrent_kernel"] = "flash-linear-attention"
    except Exception:
        report["recurrent_kernel"] = "reference pytorch"
    print(report, flush=True)

    torch.manual_seed(0)
    t = time.time()
    model = SelectorModel(SelectorConfig(text_mode=args.text_mode)).to(device)
    report["text_mode"] = args.text_mode
    report["load_s"] = round(time.time() - t, 1)
    report["params"] = {**model.param_counts(), "lora_layers": model.lora_layers}
    print("loaded:", report["params"], flush=True)

    from transformers import AutoTokenizer

    collator = Collator(AutoTokenizer.from_pretrained(model.cfg.base))
    gen = dict(max_elements=args.max_elements)

    # 1) number formats
    probe = to_device(next(synthetic_batches(collator, args.batch_size, seed=1, **gen)), device)
    modes = args.formats.split(",") if device.type == "cuda" else ["fp32"]
    report["formats"] = {m: format_check(model, probe, device, m, reps=5 if device.type == "cuda" else 2) for m in modes}
    for m, r in report["formats"].items():
        print("format", m, r, flush=True)
    good = [m for m in modes if report["formats"][m]["ok"]]
    if not good:
        raise SystemExit("no number format produced finite losses and gradients")
    amp = args.amp if args.amp != "auto" else min(good, key=lambda m: report["formats"][m]["s_per_step"])
    report["amp_used"] = amp

    # 2) can it learn? accuracy on fresh synthetic screens, before and after training
    def fresh_eval() -> dict:
        return evaluate(model, synthetic_batches(collator, args.batch_size, seed=999, **gen), args.eval_batches, device, amp)

    report["before"] = fresh_eval()
    print("before training:", report["before"], flush=True)

    ckpt = Path(tempfile.mkdtemp(prefix="bc_smoke_"))
    train_stream = synthetic_batches(collator, args.batch_size, seed=2, **gen)
    half = max(1, args.steps // 2)
    t = time.time()
    # Train in two calls on purpose: the second call must resume from the first one's checkpoint.
    train(model, train_stream, half, device, amp=amp, lr=args.lr, ckpt_dir=ckpt, save_every=half)
    history = train(model, train_stream, args.steps, device, amp=amp, lr=args.lr, ckpt_dir=ckpt, save_every=args.steps)
    report["train_s"] = round(time.time() - t, 1)
    report["resume_ok"] = bool(history and history[-1]["step"] == args.steps)
    report["loss_first"], report["loss_last"] = history[0]["loss"], history[-1]["loss"]
    report["train_tokens_per_s"] = round(statistics.median(r["tokens_per_s"] for r in history))

    report["after"] = fresh_eval()
    print("after training:", report["after"], flush=True)

    # 3) checkpoint round trip: wipe the trained parts, reload, outputs must match
    model.eval()
    with torch.no_grad(), autocast(device, amp):
        ref = model(probe)["pointer_logits"].float()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    model.load_trainable(ckpt)
    with torch.no_grad(), autocast(device, amp):
        again = model(probe)["pointer_logits"].float()
    finite = torch.isfinite(ref)
    report["checkpoint_roundtrip_max_diff"] = float((ref[finite] - again[finite]).abs().max())
    report["checkpoint_mb"] = round((ckpt / "selector.safetensors").stat().st_size / 1e6, 1)
    shutil.rmtree(ckpt, ignore_errors=True)

    # 4) how cost grows with batch size and screen size (decides the real training batch size)
    if device.type == "cuda":
        report["scaling"] = []
        for bs, n in ((8, 40), (8, 100), (4, 160)):
            b = to_device(next(synthetic_batches(collator, bs, seed=3, min_elements=n, max_elements=n)), device)
            r = format_check(model, b, device, amp, reps=2)
            report["scaling"].append({"batch": bs, "elements": n, **{k: r.get(k) for k in ("ok", "s_per_step", "tokens_per_s", "peak_mem_gb", "error") if k in r}})
            print("scaling", report["scaling"][-1], flush=True)
            del b
            torch.cuda.empty_cache()

    # 5) time per decision
    report["latency"] = [decision_latency(model, collator, device, amp, n, reps=10 if device.type == "cuda" else 3) for n in (32, 128)]

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("===REPORT===")
    print(json.dumps(report, indent=2))
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
