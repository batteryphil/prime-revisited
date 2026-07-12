import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba3_prime_native import build_prime_lut, PrimeLinear

# Patch PrimeLinear with a .weight property so HuggingFace internals
# (lm_head.weight.dtype) don't crash during CPU inference generation.
@property
def _prime_weight(self):
    combined = self.base_idx.long() * 256 + self.fine_idx.long()
    return self.lut[combined].float()

PrimeLinear.weight = _prime_weight

def replace_linear(model, lut):
    """Replace all Linear layers with PrimeLinear exactly like the training script."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parts = name.rsplit('.', 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            attr = parts[-1]
            if attr in ('in_proj', 'lm_head'):
                prime = PrimeLinear(module, lut)
                setattr(parent, attr, prime)
                replaced += 1
    return replaced

def evaluate():
    print("[INIT] Loading tokenizer and base architecture...")
    model_id = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    lut = build_prime_lut()
    replaced_count = replace_linear(model, lut)
    
    ckpt_path = "prime_mamba3_12500.pt"
    print(f"[LOAD] checkpoint path: {ckpt_path}")
    print(f"[LOAD] PRIME layers restored: {replaced_count}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Load keys
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"[LOAD] total parameters restored successfully (missing: {len(missing)}, unexpected: {len(unexpected)})")
    
    print("[INIT] Moving model to CPU for generation (GPU is busy training)...")
    model.cpu()
    model.eval()

    raw_prompts = [
        # C++ fundamentals
        "Write a C++ struct for a network packet with fields for source IP, destination IP, and payload size.",
        "Implement a C++ function to check if a system is in a SAFE or DEGRADED state based on memory usage.",
        "Write a C function using stdint.h to clamp a uint32_t value between a min and max.",
        # Operating-Organism specific
        "What is homeostasis in the context of an autonomous operating system?",
        "Write a C++ function called `check_survival_invariants` that returns true if all system vitals are nominal.",
        "Write a Rust function to monitor CPU temperature and return an enum: Normal, Warning, or Critical.",
        # Systems / baremetal
        "Write a C header file defining constants for system health states: NORMAL, DEGRADED, CRITICAL, SHUTDOWN.",
        "Implement a circular buffer in C for logging system telemetry events.",
        "Write a C++ class called `Homeostasis` with methods: stabilize(), getSurvivalScore(), and shutdown().",
        # OO Architecture
        "Describe the role of a cortex module in a baremetal operating organism architecture.",
    ]

    print("\n" + "="*60)
    print("      LIVE CPU INFERENCE (STEP 12500 CHECKPOINT)")
    print("="*60)

    for p in raw_prompts:
        alpaca_prompt = f"### Instruction:\n{p}\n### Response:\n"
        print(f"\n[PROMPT] {p}")
        input_ids = tokenizer(alpaca_prompt, return_tensors='pt').input_ids.cpu()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=150, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[OUTPUT]\n{output_text.strip()}\n{'-'*60}")

if __name__ == '__main__':
    evaluate()
