import sys
import time
import json
import torch
import fcntl
from transformers import AutoTokenizer

def generate_salad(model, tokenizer, max_new=30):
    prompt = "Q: What is the capital of France?\nA:"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    
    t0 = time.time()
    for _ in range(max_new - 1):
        with torch.no_grad():
            logits, _ = model(input_ids, domain_id=0)
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

def run():
    if len(sys.argv) < 3:
        return
        
    ckpt_path = sys.argv[1]
    step = int(sys.argv[2])
    
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    import mamba3_titan_mimo
    from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear
        
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
    
    text, tps = generate_salad(model, tokenizer)
    
    # Safely append to JSON monitor file
    json_file = "samples_titan_mimo.json"
    entry = {
        "step": step,
        "text": text,
        "tps": tps
    }
    
    # We use fcntl to lock the file since the dashboard might be reading it
    with open(json_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            data = json.load(f)
        except:
            data = []
            
        data.append(entry)
        if len(data) > 50:
            data = data[-50:]
            
        f.seek(0)
        f.truncate()
        json.dump(data, f)
        fcntl.flock(f, fcntl.LOCK_UN)
        
    # Print for the log parser
    print(f"[SAMPLE] Step {step} | TPS: {tps:.2f} | Text: {text[:30]}...")

if __name__ == '__main__':
    run()
