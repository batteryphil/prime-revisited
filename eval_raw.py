import torch
from transformers import AutoTokenizer
from mamba3_titan_mimo import TitanMIMO, CONFIG, build_prime_lut, PrimeLinear
import json
import sys

def main():
    print("Loading model...")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = TitanMIMO().to(device)
    lut = build_prime_lut()
    
    # Wrap linear layers like in offline_evaluator
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
    for arm in model.mimo_arms:
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
        
    ckpt = torch.load("/data/titan_checkpoints/titan_mimo_live.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print("Extracting raw samples from mega corpus...")
    tool_sample = None
    code_sample = None
    
    with open('/data/titan_mega_corpus.jsonl') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('domain') == 'tool' and tool_sample is None and "<tool_call>" in obj['text']:
                tool_sample = obj['text']
            if obj.get('domain') == 'code' and code_sample is None:
                code_sample = obj['text']
            if tool_sample and code_sample:
                break
                
    # --- EVAL CODE ---
    print("\n" + "="*50)
    print(" RAW CODE EVALUATION")
    print("="*50)
    # Split at Output: 
    code_prompt = code_sample.split("Output:")[0] + "Output:"
    print("--- PROMPT ---")
    print(code_prompt)
    print("--- GENERATING ---")
    
    # Tokenize but get domain_id
    inputs = tokenizer(code_prompt, return_tensors="pt").input_ids.to(device)
    domain_id = 2
    
    gen_ids = inputs[0].tolist()
    for _ in range(100):
        with torch.no_grad():
            logits, _ = model(torch.tensor([gen_ids], device=device), domain_id=domain_id)
            next_token = logits[0, -1].argmax().item()
            gen_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
    print(tokenizer.decode(gen_ids[len(inputs[0]):], skip_special_tokens=True))
    
    # --- EVAL TOOL ---
    print("\n" + "="*50)
    print(" RAW TOOL EVALUATION")
    print("="*50)
    # Split at ASSISTANT:
    tool_prompt = tool_sample.split("ASSISTANT:")[0] + "ASSISTANT:"
    print("--- PROMPT ---")
    print(tool_prompt)
    print("--- GENERATING ---")
    
    inputs = tokenizer(tool_prompt, return_tensors="pt").input_ids.to(device)
    domain_id = 3
    
    gen_ids = inputs[0].tolist()
    for _ in range(100):
        with torch.no_grad():
            logits, _ = model(torch.tensor([gen_ids], device=device), domain_id=domain_id)
            next_token = logits[0, -1].argmax().item()
            gen_ids.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
    print(tokenizer.decode(gen_ids[len(inputs[0]):], skip_special_tokens=True))

if __name__ == "__main__":
    main()
