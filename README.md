# Titan MIMO PRIME
*A 650M-parameter Mamba language model trained with a discrete vote-based optimizer and autonomous mixture-of-experts routing.*

## Research Status
> [!WARNING]
> This repository is experimental research.
> The architecture is actively being trained and evaluated. The optimization algorithm is stable but remains under investigation. Benchmarking against conventional AdamW-trained Mamba baselines is ongoing.

---

## What We Are Doing
This repository contains the architecture and training pipeline for **Titan MIMO PRIME**, a ~650 million parameter language model built entirely from scratch in PyTorch. It represents a radical departure from traditional architectures and standard continuous optimization techniques. 

Instead of Transformers and AdamW, this model uses a Mamba S6 backbone, hard-routed Mixture-of-Experts (MoE), and a purely discrete optimization paradigm.

## Why Discrete Optimization?
Conventional optimizers update billions of floating-point weights every training step by continuously subtracting gradients: `weight -= lr * gradient`.

PRIME instead treats optimization as a discrete consensus process. PRIME accumulates directional evidence over multiple microbatches. Only after sufficient consensus does a parameter move to a neighboring location on a structured lookup manifold.

This provides:
*   Bounded updates
*   Deterministic parameter motion
*   Inherent quantization
*   Resistance to transient gradient noise

---

## The PRIME Optimizer

```text
       Gradient
          │
          ▼
        sign()
          │
          ▼
 Vote Buffer (int16)
          │
          ▼
    PI Controller
          │
          ▼
  Threshold Passed?
          │
      ┌───┴────┐
      │        │
     No       Yes
      │        │
      ▼        ▼
    stay   index += ±1
               │
               ▼
       Prime Harmonic LUT
               │
               ▼
          Weight Value
```

The core `in_proj` and `out_proj` weight matrices of every Mamba layer are wrapped in a custom `PrimeLinear` module. 
- All weight matrices live as **16-bit indices** into a 65,536-entry **Prime Harmonic Grid** LUT.
- The LUT is interpolated between prime reciprocal anchors: `[±1/2, ±1/3, ±1/5 ... ±2.0]`.
- Each backward pass accumulates `sign(grad) × 10` into a **GPU vote buffer** (`int16`).
- Blocks of weights that exceed a dynamic supermajority threshold have their index stepped `±1` across the manifold.

**There is no AdamW. There is no SGD. The vote mechanism IS the optimizer.**

---

## Autonomous PI Controller (Homeostasis)
Every layer regulates its own learning rate indirectly.

Rather than using a fixed update threshold, each layer continuously adjusts the amount of evidence required before a discrete weight update occurs using a **Proportional-Integral (PI) Controller**.

Layers that become unstable automatically become more conservative (the log-threshold raises). Layers that stagnate automatically become more plastic (the log-threshold lowers). This creates a homeostatic learning process similar to adaptive control systems, preventing early layers from freezing and late layers from exploding.

---

## Architecture: Titan MIMO

```text
                 Tokens
                    │
                    ▼
             Token Embedding
                    │
                    ▼
         Shared Mamba Trunk
            Layers 0-13
                    │
                    ▼
          LayerNorm (detached)
                    │
                    ▼
          Domain Router (PRIME)
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     Chat         Math         Code  ... (and Tool)
       │            │            │
       └────────────┬────────────┘
                    ▼
             Shared Crown
            Layers 14-27
                    │
                    ▼
                LM Head
```

To prevent "catastrophic forgetting" across widely different domains, the model is bisected.
*   **Shared Trunk:** Processes raw token syntax generically.
*   **Dynamic Domain Router:** A discrete linear router analyzes the context. Using a Gumbel-Softmax Straight-Through Estimator during training (and greedy argmax during inference), it autonomously routes the forward pass.
*   **Expert Arms:** 4 distinct Mamba blocks (Chat, Math, Code, Tool) specializing deeply in their respective latent geometries.
*   **Shared Crown:** Collapses expert output back into the residual stream.

---

## Feature Comparison

| Feature | Transformer | Standard Mamba | Titan PRIME |
| :--- | :---: | :---: | :---: |
| Attention | ✓ | ✗ | ✗ |
| State Space | ✗ | ✓ | ✓ |
| MoE | Optional | Rare | ✓ |
| AdamW | ✓ | ✓ | ✗ |
| Discrete Optimizer | ✗ | ✗ | ✓ |
| PI Homeostasis | ✗ | ✗ | ✓ |

---

## Preliminary Benchmarks
*   **Parameters:** ~650M
*   **Hardware:** RTX 3060 (12GB VRAM)
*   **Throughput:** ~670 Tokens/sec
*   **Routing Accuracy:** ~98% (Router flip rate stable at 2.08%)
*   **Training Status:** Active (Currently at Step ~345,000 / 510,000)

*(Final Validation and Perplexity benchmarks will be posted upon training completion).*

---

## Telemetry & Observability
The training loop exports high-resolution JSON telemetry to `telemetry_live.json` on every interval. A local dashboard parses this for easy monitoring.

*(Dashboard screenshot placeholder - to be uploaded)*

---

## How to Run
This repository is configured to be reproducible. Ensure you have PyTorch, Mamba-SSM, and Triton installed via the `requirements.txt`.

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Configuration
The scripts look for two `.jsonl` files (training corpus and validation suite).
*   Format: `{"text": "Your sequence here...", "domain": "code"}`

### 3. Start Training
The `mamba3_titan_mimo.py` script acts as the main training loop. It will automatically load the latest checkpoint from your `ckpt_dir` if one exists.
```bash
nohup python3 mamba3_titan_mimo.py \
    --corpus_path ./data/titan_mega_corpus.jsonl \
    --val_path ./data/titan_val_suite.jsonl \
    --ckpt_dir ./checkpoints \
    > output.log 2>&1 &
```

### 4. Monitor Telemetry
```bash
python3 serve.py
```

### 5. Evaluate Autonomous Routing
```bash
python3 eval_raw.py --ckpt_path ./checkpoints/titan_mimo_live.pt --corpus_path ./data/titan_mega_corpus.jsonl
```

---

## Future Work
*   Multi-expert routing (top-k)
*   Hierarchical routers
*   Learned LUT geometry (evolving the prime fractions)
*   Adaptive flip-rate targets
*   Dynamic expert expansion
*   Distributed PRIME training across multi-GPU
*   Continuous-discrete hybrid projections
