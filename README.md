# Mamba3-300M PRIME — Training from Scratch on the Prime Harmonic Grid

## What We Are Doing

Building on the results in [prime-revisited](https://github.com/batteryphil/prime-revisited) (which proved PRIME works on Mamba-130m fine-tuning), we are now **training a 300M Mamba3 from scratch** using the same discrete vote-based optimizer — no AdamW, no SGD, no continuous weight updates at all.

The hypothesis: if PRIME can drive directed optimization when *converting* a pretrained model, it should also be capable of driving learning from a *cold random initialization* on the prime harmonic grid.

---

## The PRIME Method (from prime-revisited)

Standard training: `gradient → float weight subtraction`

PRIME training: `gradient sign → vote accumulation → supermajority gate → LUT index step`

- All weight matrices live as **16-bit indices** into a 65,536-entry **Prime Harmonic Grid** LUT
- The LUT is interpolated between prime reciprocal anchors: `[±1/2, ±1/3, ±1/5, ±1/7 ... ±2.0]`
- Each backward pass accumulates `sign(grad) × 10` into a **CPU vote buffer** (no VRAM cost)
- After `grad_accum` steps, blocks of 32 weights that exceed the **SUPERMAJORITY threshold (5)** have their index stepped `±1` across the manifold
- Momentum: consecutive same-direction steps increase stride (up to 4×); reversals get stride=0

**There is no AdamW. There is no SGD. The vote mechanism IS the optimizer.**

---

## Architecture: Mamba3-300M from Scratch

| Config | Value |
|---|---|
| `d_model` | 1024 |
| `n_layers` | 28 |
| `expand` | 2 (d_inner = 2048) |
| `d_state` | 16 |
| `seq_len` | 256 |
| `grad_accum` | 4 |
| Vocab | GPT-NeoX 50k |
| **Total params** | **~290M** |

**SSM implementation:** Pure-PyTorch `RealMambaSSM` — vectorized parallel cumsum scan in float32 (avoids bfloat16 exp() overflow). Uses `torch.utils.checkpoint` per layer.

**PrimeLinear wrapping:** `in_proj` and `out_proj` in every MambaLayer (56 layers total). All other layers (LayerNorm, SSM dynamics, embedding, lm_head) stay as standard `nn.Linear`/`nn.Embedding` with fixed init weights for this experiment.

---

## Key Files

| File | Purpose |
|---|---|
| `mamba3_prime_native.py` | **Main training script** — run this |
| `serve.py` | Dashboard server (reads `stats_mamba3_native.json`) |
| `stats_mamba3_native.json` | Live telemetry written each step |
| `training_mamba3_native.log` | Full training log |

---

## How to Run

```bash
# Activate the venv
source ~/.gemini/antigravity/scratch/analysis_project/titan_venv/bin/activate

# Start training (background)
cd /home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime
nohup python3 mamba3_prime_native.py > output.log 2> error.log &

# Monitor
tail -f training_mamba3_native.log

# Dashboard
python3 serve.py
```

---

## Telemetry Fields

| Field | Meaning |
|---|---|
| `Loss` | Cross-entropy on next-token prediction |
| `Mig %` | % of weight indices that have moved from their initial position |
| `Disp95` | 95th percentile LUT index displacement from init |
| `Ent` | Shannon entropy over occupied LUT bins (higher = more spread) |
| `TPS` | Tokens per second throughput |

---

## Proven Results (from prime-revisited baseline)

On Mamba-130m conversion experiment:
- Scale-only baseline (frozen discrete): Loss **stalled** at ~1193 after 25 steps
- PRIME system (votes active): Loss dropped to **937.5** at step 24, Migration=19.3%, Disp95=172.9

**PRIME wins decisively.** This experiment extends the test to cold-start 300M scale.

---

## Environment

- **GPU:** RTX 3060 12GB
- **PyTorch:** 2.5.1+cu121
- **Python:** 3.12
- **venv:** `~/scratch/analysis_project/titan_venv`
- **Dataset:** `bigcode/the-stack-smol` (Python subset) + local OO code chunks
