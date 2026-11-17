"""Training loop with checkpoint and resume, built for sessions that can end at any time."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterator

import torch

from .model import SelectorModel

AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast(device: torch.device, amp: str):
    dtype = AMP_DTYPES[amp]
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


@torch.no_grad()
def evaluate(model: SelectorModel, batches: Iterator[dict], n_batches: int, device: torch.device, amp: str) -> dict[str, float]:
    model.eval()
    n = ptr_ok = act_ok = 0
    none_n = none_ok = 0
    for _ in range(n_batches):
        batch = to_device(next(batches), device)
        with autocast(device, amp):
            out = model(batch)
        pred = out["pointer_logits"].argmax(dim=1)
        ptr_ok += (pred == batch["target"]).sum().item()
        act_ok += (out["action_logits"].argmax(dim=1) == batch["action"]).sum().item()
        is_none = batch["target"] == 0
        none_n += is_none.sum().item()
        none_ok += (pred[is_none] == 0).sum().item()
        n += batch["target"].shape[0]
    model.train()
    return {
        "pointer_acc": ptr_ok / n,
        "action_acc": act_ok / n,
        "none_acc": none_ok / max(1, none_n),
        "samples": n,
    }


def train(
    model: SelectorModel,
    batches: Iterator[dict],
    steps: int,
    device: torch.device,
    amp: str = "fp32",
    lr: float = 2e-4,
    warmup: int = 20,
    ckpt_dir: str | Path | None = None,
    save_every: int = 200,
    log_every: int = 20,
) -> list[dict]:
    """Runs up to `steps` optimizer steps. If ckpt_dir holds a checkpoint, continues from it."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=amp == "fp16")
    step, history = 0, []

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_dir and (ckpt_dir / "state.pt").exists():
        state = torch.load(ckpt_dir / "state.pt", map_location="cpu")
        model.load_trainable(ckpt_dir)
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step, history = state["step"], state["history"]
        print(f"resumed from step {step}", flush=True)

    def save() -> None:
        if not ckpt_dir:
            return
        model.save_trainable(ckpt_dir)
        tmp = ckpt_dir / "state.pt.tmp"
        torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "history": history}, tmp)
        tmp.replace(ckpt_dir / "state.pt")
        (ckpt_dir / "history.json").write_text(json.dumps(history))

    model.train()
    t0, tokens, logged_at = time.time(), 0, step
    while step < steps:
        batch = to_device(next(batches), device)
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        with autocast(device, amp):
            out = model(batch)
        loss = model.loss(out, batch)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at step {step} with amp={amp}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens += int(batch["intent_mask"].sum() + batch["elem_mask"].sum()) + batch["target"].shape[0]

        if step % log_every == 0 or step == steps:
            dt = time.time() - t0
            row = {"step": step, "loss": round(loss.item(), 4), "tokens_per_s": round(tokens / dt), "s_per_step": round(dt / (step - logged_at), 3)}
            history.append(row)
            print(row, flush=True)
            t0, tokens, logged_at = time.time(), 0, step
        if ckpt_dir and (step % save_every == 0 or step == steps):
            save()
    return history
