import sys
import time
import torch
from transformers import AutoTokenizer

import mamba3_titan_mimo
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear

def generate_text(model, tokenizer, prompt, domain_id, max_new=50):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    
    t0 = time.time()
    for _ in range(max_new - 1):
        with torch.no_grad():
            logits, _ = model(input_ids, domain_id=domain_id)
            logits = logits[:, -1, :].float()
            
        temperature = 0.8
        top_k = 50
        logits = logits / max(temperature, 1e-8)
        
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        if next_id.item() == tokenizer.eos_token_id:
            break
            
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        
    elapsed = time.time() - t0
    tps = max_new / max(elapsed, 1e-6)
    
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, tps

if __name__ == '__main__':
    ckpt_path = '/data/titan_checkpoints/titan_mimo_live.pt'
    print(f"Loading checkpoint {ckpt_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = TitanMIMO()
    lut = build_prime_lut()
    wrapped = 0
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
        wrapped += 2
    for arm in model.mimo_arms:
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
        wrapped += 2
        
    ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt_data['state_dict'], strict=False)
    model.cuda().eval()
    
    print("Model loaded on GPU. Running comprehensive evaluation suite...\n")
    
    tests = [
        {"domain": 0, "name": "CHAT", "prompt": "### Instruction:\nHello! Who are you and what can you do?\n### Response:\n"},
        {"domain": 1, "name": "MATH", "prompt": "### Instruction:\nCalculate 15 + 27.\n### Response:\n"},
        {"domain": 2, "name": "CODE", "prompt": "### Instruction:\nWrite a Python function to calculate the nth Fibonacci number.\n### Response:\n"},
        {"domain": 3, "name": "TOOL", "prompt": "### System (Tool Config):\nYou have access to a web_search tool.\n### Conversation:\nUser: Search the web for artificial intelligence news.\nAgent:"}
    ]
    
    for t in tests:
        print(f"============================================================")
        print(f" DOMAIN: {t['name']} (ID: {t['domain']})")
        print(f"============================================================")
        text, tps = generate_text(model, tokenizer, t['prompt'], t['domain'])
        print(f"TPS: {tps:.2f}")
        print(f"Output:\n{text}\n")
