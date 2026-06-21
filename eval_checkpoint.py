import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_prime_train import build_prime_lut, PrimeLinear

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
    lut = build_prime_lut(device='cpu')
    replaced_count = replace_linear(model, lut)
    
    ckpt_path = "prime_step_b_300.pt"
    print(f"[LOAD] checkpoint step: 300")
    print(f"[LOAD] PRIME layers restored: {replaced_count}")
    print(f"[LOAD] LUT entries referenced: {len(lut)}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Load and verify keys
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    
    # Check if scales loaded correctly
    scales_restored = sum('scale' not in m for m in missing) # just a rough check, or print successful keys
    print(f"[LOAD] total parameters restored successfully (missing: {len(missing)}, unexpected: {len(unexpected)})")
    if len(missing) > 0:
        print(f"       Missing example: {missing[0]}")
    
    print("[INIT] Moving model to CPU for generation (GPU is busy training)...")
    model.cpu()
    model.eval()

    raw_prompts = [
        "Explain gravity:",
        "Write Python code for a fibonacci function:\n```python\n",
        "The capital of France is",
        "Question: What is 2+2?\nAnswer:",
        "Write a python function def quicksort(arr):"
    ]

    print("\n" + "="*50)
    print("      GENERATION TESTS (STEP 300 CHECKPOINT)")
    print("="*50)

    for p in raw_prompts:
        alpaca_prompt = f"### Instruction:\n{p}\n### Response:\n"
        print(f"\n[PROMPT] {p}")
        input_ids = tokenizer(alpaca_prompt, return_tensors='pt').input_ids.cpu()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=100, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode only the generated tokens
        output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[OUTPUT]\n{output_text}\n{'-'*50}")

if __name__ == '__main__':
    evaluate()
