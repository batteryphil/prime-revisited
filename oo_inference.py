"""
oo_inference.py
Run a full battery of OO/C++ test prompts through the PRIME Mamba checkpoint
on CPU, using the NATIVE Mamba3LM architecture from the training script.
"""

import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer
# Import the exact same model class and helpers used during training
from mamba3_prime_native import build_prime_lut, PrimeLinear, Mamba3LM

# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    model_id = "state-spaces/mamba-130m-hf"
    print(f"[LOAD] Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size

    # Peek at the checkpoint to get the exact vocab size used during training
    import os
    ckpt_peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    vocab_size = ckpt_peek['state_dict']['embedding.weight'].shape[0]
    print(f"[LOAD] Vocab size from checkpoint: {vocab_size}")

    # Build the exact same architecture used during training
    print(f"[LOAD] Instantiating native Mamba3LM (vocab={vocab_size})...")
    model = Mamba3LM(vocab_size)

    missing, unexpected = model.load_state_dict(ckpt_peek['state_dict'], strict=False)
    print(f"[LOAD] {ckpt_path} — step {ckpt_peek['step']}, missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.cpu().eval()
    return model, tokenizer

# ── Manual token-by-token generation ─────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=120, temperature=0.8, top_k=50):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    for _ in range(max_new):
        # Native Mamba3LM returns (logits, loss) — unpack accordingly
        logits, _ = model(input_ids)
        logits = logits[:, -1, :].float()          # last token logits

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Stop on EOS
        if next_id.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_id], dim=-1)

    # Decode only the newly generated tokens
    new_ids = input_ids[0, tokenizer(prompt, return_tensors='pt').input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

# ── Test Battery ──────────────────────────────────────────────────────────────
PROMPTS = [
    # C++ fundamentals
    ("C++ struct for network packet",
     "Write a C++ struct for a network packet with fields for source IP, destination IP, and payload size."),
    ("SAFE/DEGRADED state check",
     "Implement a C++ function to check if a system is in a SAFE or DEGRADED state based on memory usage."),
    ("uint32_t clamp in C",
     "Write a C function using stdint.h to clamp a uint32_t value between a min and max."),
    # Operating-Organism specific
    ("Homeostasis in an OS",
     "What is homeostasis in the context of an autonomous operating system?"),
    ("check_survival_invariants()",
     "Write a C++ function called check_survival_invariants that returns true if all system vitals are nominal."),
    ("Rust CPU temp monitor",
     "Write a Rust function to monitor CPU temperature and return an enum: Normal, Warning, or Critical."),
    # Systems / baremetal
    ("C header: health state constants",
     "Write a C header file defining constants for system health states: NORMAL, DEGRADED, CRITICAL, SHUTDOWN."),
    ("Circular telemetry buffer",
     "Implement a circular buffer in C for logging system telemetry events."),
    ("Homeostasis C++ class",
     "Write a C++ class called Homeostasis with methods: stabilize(), getSurvivalScore(), and shutdown()."),
    # OO Architecture
    ("Role of cortex module",
     "Describe the role of a cortex module in a baremetal operating organism architecture."),
]

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import glob, os
    # Auto-pick latest checkpoint
    ckpts = sorted(glob.glob('prime_mamba3_*.pt'),
                   key=lambda f: int(f.split('_')[-1].replace('.pt', '')))
    ckpt = ckpts[-1]

    model, tokenizer = load_model(ckpt)
    print(f"\n{'='*65}")
    print(f"  OO / C++ INFERENCE BATTERY  —  {ckpt}")
    print(f"{'='*65}")

    results = []
    for label, prompt in PROMPTS:
        alpaca = f"### Instruction:\n{prompt}\n### Response:\n"
        t0 = time.time()
        output = generate(model, tokenizer, alpaca)
        elapsed = time.time() - t0
        tps = 120 / elapsed  # approx
        print(f"\n[{label}]")
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT:\n{output.strip()}")
        print(f"--- ({elapsed:.1f}s, ~{tps:.1f} TPS) ---")
        results.append((label, prompt, output, elapsed))

    # Summary
    avg_tps = 120 / (sum(r[3] for r in results) / len(results))
    print(f"\n{'='*65}")
    print(f"  COMPLETE — {len(results)} prompts, avg TPS: {avg_tps:.2f}")
    print(f"{'='*65}")
