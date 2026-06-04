# tiny-dllm

A Masked Diffusion Language Model built from scratch in PyTorch — for learning and robotics research.

## What is a Diffusion Language Model?

Unlike GPT-style models that generate text left-to-right one token at a time,
a dLLM starts with a fully masked sequence and iteratively **denoises** it — revealing
tokens in parallel based on confidence. This enables bidirectional context and
non-sequential generation.

```
[MASK][MASK][MASK][MASK][MASK]   ← start (fully masked, no input needed)
[MASK] the [MASK] fox [MASK]     ← step 3
 The   the quick fox jumps       ← step 10 (done)
```

## Files

| File | What you learn |
|---|---|
| `01_tensors.py` | PyTorch tensors, autograd, nn.Module, training loop |
| `02_attention.py` | Multi-head self-attention from scratch |
| `03_transformer.py` | Full transformer backbone (~10M params) |
| `04_diffusion.py` | Masked diffusion — forward noise + denoising sampler |
| `05_train.py` | Train on TinyShakespeare |
| `06_generate.py` | Generate text from a trained checkpoint |

## Setup

```bash
pip install torch numpy matplotlib tqdm
```

Requires Python 3.10+ and PyTorch 2.0+. GPU recommended (RTX 3050 works great).

## Run

```bash
# Learn step by step — read each file, then run it
python 01_tensors.py
python 02_attention.py
python 03_transformer.py
python 04_diffusion.py

# Train (~15-30 mins on RTX 3050, downloads Shakespeare automatically)
python 05_train.py

# Generate — no input needed, model generates on its own
python 06_generate.py
python 06_generate.py --steps 30 --len 200 --temp 0.8
```

## Model Architecture

```
Token IDs [B, T]
    ↓ Embedding (256-dim)
[B, T, 256]
    ↓ × 4 Transformer Blocks
       └─ LayerNorm → Multi-Head Attention (4 heads) → residual
       └─ LayerNorm → FFN (256→1024→256, GELU) → residual
[B, T, 256]
    ↓ LayerNorm → Linear → vocab_size
Logits [B, T, vocab_size]
```

~10M parameters. Trains on TinyShakespeare (~1MB). Loss drops from ~4.2 → ~1.4 over 5000 steps.

## Roadmap

- [x] Character-level tokenizer
- [x] Transformer backbone
- [x] Masked diffusion training
- [x] Iterative confidence-based sampling
- [ ] Fill-in-the-blanks (conditional generation)
- [ ] Fine-tune on robot task sequences (ROS2 action plans)
- [ ] Conditional generation (robot state → action plan)
