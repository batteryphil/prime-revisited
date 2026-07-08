# Titan MIMO 650M — Autonomous Discrete MoE Mamba

## What We Are Doing
This repository contains the architecture and training pipeline for the **Titan MIMO 650M**, a ~650 million parameter language model built entirely from scratch in PyTorch. It represents a radical departure from traditional architectures and standard continuous optimization techniques. 

1.  **Architecture:** It utilizes the Mamba S6 (Selective State Space) algorithm rather than attention, allowing linear-time sequence processing. Furthermore, it employs a Multi-Input Multi-Output (MIMO) Mixture-of-Experts architecture, branching the backbone into four distinct expert arms (Chat, Math, Code, Tool).
2.  **Optimization:** The network completely abandons FP32 continuous gradient descent (no AdamW/SGD) for its core linear projections. Instead, it utilizes the **PRIME Optimizer**, a purely discrete, vote-based mechanism that maps weights to a static Prime Harmonic Look-Up Table (LUT).

---

## 1. The PRIME Optimizer (ContinuousVoteOptimizer)
Standard training: `gradient → float weight subtraction`
PRIME training: `gradient sign → vote accumulation → PI Controller threshold → LUT index step`

The core `in_proj` and `out_proj` weight matrices of every Mamba layer are wrapped in a custom `PrimeLinear` module. 
- All weight matrices live as **16-bit indices** into a 65,536-entry **Prime Harmonic Grid** LUT.
- The LUT is interpolated between prime reciprocal anchors: `[±1/2, ±1/3, ±1/5 ... ±2.0]`.
- Each backward pass accumulates `sign(grad) × 10` into a **GPU vote buffer** (`int16`).
- Blocks of weights that exceed a dynamic supermajority threshold have their index stepped `±1` across the manifold.

**There is no AdamW. There is no SGD. The vote mechanism IS the optimizer.**

---

## 2. Autonomous PI Controller (Homeostatic Volatility)
In a 650M parameter model, layers experience vastly different gradient variances. A static supermajority threshold causes early layers to freeze and late layers to become stochastic noise generators.

To fix this, every single `PrimeLinear` layer possesses an autonomous **Proportional-Integral (PI) Controller**.
*   The user sets a `target_flip_rate` (e.g., 10% for backbone layers, 16% for the domain router).
*   Every 50 steps, the PI controller calculates the layer's Exponential Moving Average (EMA) flip rate.
*   It compares the EMA to the target rate and dynamically adjusts the supermajority log-threshold (using $K_p=0.5, K_i=0.01$). 
*   If a layer flips too aggressively, the PI controller raises the threshold, mathematically forcing the layer to require stronger gradient consensus before altering weights. This absorbs massive gradient shocks without catastrophic forgetting.

---

## 3. Architecture: Titan MIMO
To prevent "catastrophic forgetting" across widely different domains (e.g., Python syntax overriding Conversational English syntax), the model is bisected.

*   **Shared Trunk (Layers 0-13):** The first half of the Mamba backbone processes raw token syntax generically.
*   **Dynamic Domain Router:** At the mid-point, a detached `F.layer_norm` bounds the residual stream, and a discrete linear router analyzes the context. Using a Gumbel-Softmax Straight-Through Estimator during training (and greedy argmax during inference), it autonomously decides which expert arm is required.
*   **Expert Arms (Chat, Math, Code, Tool):** The forward pass is routed into one of four distinct Mamba S6 blocks. These blocks specialize deeply in their respective latent geometries.
*   **Shared Crown (Layers 14-27):** The expert arm output is collapsed back into the residual stream and processed by the final generic reasoning layers.

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
The scripts look for two `.jsonl` files (training corpus and validation suite). You can place these in a `./data/` directory or pass them via CLI. 
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
The training loop exports high-resolution JSON telemetry to `telemetry_live.json` on every interval. A local dashboard parses this for easy monitoring.
```bash
# Start the local dashboard server
python3 serve.py
```

### 5. Evaluate Autonomous Routing
To evaluate the model's autonomous routing capabilities and raw text generation (without AdamW interference), run the evaluation script:
```bash
python3 eval_raw.py --ckpt_path ./checkpoints/titan_mimo_live.pt --corpus_path ./data/titan_mega_corpus.jsonl
```

---

## Hardware Environment
*   **GPU Required:** NVIDIA GPU with CUDA support (Triton and Mamba-SSM kernels require modern CUDA architectures). 
*   **Development Rig:** RTX 3060 (12GB VRAM). The training pipeline is highly optimized (Batch Size 4, Grad Accumulation 2) to fit a 650M MoE model inside 12GB of VRAM while maintaining ~670 Tokens Per Second throughput.
