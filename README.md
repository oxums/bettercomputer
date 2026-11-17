# bettercomputer

A small, fast executor model for computer use. A large model plans in intents ("press the buy button"); this model looks at the elements currently on screen and picks which one to act on and how, in tens of milliseconds, so the large model never sits in the per-click loop.

## What is here

- `bettercomputer/model.py`: the selector model. Qwen3.5-0.8B-Base with LoRA, an element adapter (a screen element's box and type become one input vector, followed by its text tokens), a pointer head (picks an element or "none of these") and an action head.
- `bettercomputer/data.py`: the sample format, batching, and a synthetic pointing task.
- `bettercomputer/train.py`: training loop with checkpoint and resume.
- `bettercomputer/smoke.py`: stage 0 smoke test.
- `notebooks/00_smoke_test.ipynb`: runs the smoke test on a Colab T4.

## Status

Stage 0 (smoke test) has run on a Colab T4. No real data yet; the model has only been trained on a synthetic pointing task.

| Check | Result |
|---|---|
| Loads and trains on a T4 | Yes. fp16 is fastest (0.77 s/step), then fp32 (1.01 s), bf16 (1.62 s), batch of 16 short screens |
| Checkpoint save, reload, resume | Exact (52 MB per checkpoint) |
| Action accuracy, 500 steps | 100% |
| Pointing at on-screen targets, text as tokens | 0% to 80% in 500 steps (4,000 samples) |
| Pointing at on-screen targets, one vector per element | 19% in 500 steps (8,000 samples) |
| Training memory | about 3.4 MB per token on top of 3 GB for the base model |

Feeding each element's text as ordinary tokens is now the default. It costs several positions per element instead of one.

Colab notes: the recurrent-layer kernel (`flash-linear-attention`, already present in the Colab runtime) compiles for about 5 minutes per number format on first use in a fresh runtime. `peft` is not used because it refuses to load next to Colab's `torchao`.

## Run the smoke test

On Colab: open `notebooks/00_smoke_test.ipynb`, pick a T4 GPU runtime, run all.

Locally (CPU works, slowly):

```bash
uv venv && uv pip install -e .
python -m bettercomputer.smoke --steps 6 --batch-size 4 --eval-batches 2 --max-elements 12
```
