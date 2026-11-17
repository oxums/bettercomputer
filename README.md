# bettercomputer

A small, fast executor model for computer use. A large model plans in intents ("press the buy button"); this model looks at the elements currently on screen and picks which one to act on and how, in tens of milliseconds, so the large model never sits in the per-click loop.

## What is here

- `bettercomputer/model.py`: the selector model. Qwen3.5-0.8B-Base with LoRA, an element adapter (one screen element becomes one input vector), a pointer head (picks an element or "none of these") and an action head.
- `bettercomputer/data.py`: the sample format, batching, and a synthetic pointing task.
- `bettercomputer/train.py`: training loop with checkpoint and resume.
- `bettercomputer/smoke.py`: stage 0 smoke test.
- `notebooks/00_smoke_test.ipynb`: runs the smoke test on a Colab T4.

## Status

Stage 0 (smoke test). No real data yet; the model has only been trained on the synthetic task.

## Run the smoke test

On Colab: open `notebooks/00_smoke_test.ipynb`, pick a T4 GPU runtime, run all.

Locally (CPU works, slowly):

```bash
uv venv && uv pip install -e .
python -m bettercomputer.smoke --steps 6 --batch-size 4 --eval-batches 2 --max-elements 12
```
