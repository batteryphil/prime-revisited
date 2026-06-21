# Continuous PRIME: Applying Vote-Gate Optimization to FP32/FP16

## 1. The Core Premise

The Phase 2 cold-start experiment strongly suggests that the actual engine of learning in the PRIME architecture is not the prime harmonic Look-Up Table (LUT) itself, but the **discrete, vote-gated update rule**.

Currently, the rule is:
1. Extract gradient sign.
2. Accumulate in an `int16` vote buffer.
3. If `abs(vote) > SUPERMAJORITY`: step the discrete index $\pm 1$.

This raises an immediate follow-up hypothesis: **Can the voting mechanism be decoupled from the prime manifold and applied directly to standard continuous `fp32`/`fp16` weights?**

## 2. Mechanics of Continuous PRIME

In a Continuous PRIME setup, the network architecture is entirely standard (e.g., standard `nn.Linear` layers, no LUT, no index tracking).

**The Optimizer State:**
Instead of storing high-precision continuous momentum and variance states (as AdamW does, costing 8 bytes per parameter), Continuous PRIME uses a highly quantized **`int8` vote buffer** (costing 1 byte per parameter).

**The Update Rule:**
```python
# 1. Accumulate directional pressure
vote_buffer += torch.sign(grad).to(torch.int8)

# 2. Gate updates based on consensus
authorized = abs(vote_buffer) > SUPERMAJORITY

# 3. Apply sparse, continuous update
step_direction = torch.sign(vote_buffer[authorized])
weight[authorized] += step_direction * STEP_SIZE

# 4. State management
vote_buffer[authorized] = 0
vote_buffer[~authorized] = (vote_buffer[~authorized] * DECAY_RATE).round()
```

## 3. Theoretical Advantages over AdamW

If this mechanism converges competitively with AdamW, it offers massive architectural advantages:

1. **Extreme Memory Efficiency**:
   - AdamW overhead: 8 bytes / parameter (fp32 $m$ and $v$).
   - Continuous PRIME overhead: 1 byte / parameter (int8 vote).
   - *Result*: An **87.5% reduction** in optimizer VRAM overhead, allowing significantly larger models or batch sizes on constrained hardware.

2. **Sparse Compute**:
   - AdamW updates 100% of the network's parameters on every step.
   - A vote-gated system with `SUPERMAJORITY = 12` might only update 20-30% of parameters per step, skipping the memory write operations for the remaining 70-80%.

3. **Inherent Noise Immunity**:
   - Mini-batch gradients are inherently noisy. AdamW smooths this with exponential moving averages, but every noisy batch still moves the weight slightly.
   - Continuous PRIME demands a strict temporal consensus. If a weight does not receive sustained pressure in a consistent direction across multiple batches, it does not move at all.

## 4. The Step Size Problem

The primary challenge of mapping PRIME to continuous space is defining the `STEP_SIZE`. In the prime harmonic LUT, the step sizes are naturally non-linear (the delta between 1/17 and 1/19 is much smaller than between 1.0 and 1.5).

In continuous space, we must decide how far to move the weight once the gate opens. Potential implementations:

- **Absolute Stride**: `STEP_SIZE = LR` (Constant scalar addition. Likely too rigid).
- **Proportional Stride**: `STEP_SIZE = abs(weight) * LR` (Weight scales by a percentage. Behaves similarly to weight decay bounds).
- **Dynamic Momentum**: The step size scales with the *consecutive* number of times the block has moved in the same direction, recreating the `MAX_STRIDE` momentum acceleration from the current PRIME implementation.

## 5. Proposed Future Experiment

To rigorously isolate the variable of the voting mechanism, a future experiment should:
1. Take the exact `mamba3_adamw_baseline.py` script.
2. Replace `torch.optim.AdamW` with a custom `ContinuousPrimeOptimizer` subclassing `torch.optim.Optimizer`.
3. Train from scratch and compare the loss curve against both the AdamW baseline and the discrete Prime Harmonic run.

If Continuous PRIME successfully learns, it proves that the temporal consensus vote-gate is a viable alternative to continuous gradient descent in modern deep learning.
