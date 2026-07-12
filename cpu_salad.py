import sys
import time
import json
import torch
import fcntl
from transformers import AutoTokenizer

sys.path.append('/home/phil/.gemini/antigravity/scratch/baremetal_test/harmonic_convergence')
import mamba3_titan_mimo
from mamba3_prime_native import RealMambaSSM
# Monkeypatch the CUDA Mamba kernel to the pure PyTorch CPU equivalent
mamba3_titan_mimo.Mamba = RealMambaSSM
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear

def generate_salad(model, tokenizer, max_new=30):
    prompt = "Q: What is the capital of France?\nA:"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cpu')
    
    t0 = time.time()
    for step in range(max_new - 1):
        with torch.no_grad():
            try:
                logits, _ = model(input_ids, domain_id=0)
            except Exception as e:
                print(f"Error at step {step}: {e}")
                break
                
            if torch.isnan(logits).any():
                print(f"NaN encountered at step {step}")
                break
                
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
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/data/titan_checkpoints/titan_mimo_live.pt'
    
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
        
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt_data['state_dict'], strict=False)
    model.cpu().eval()
    
    print("Generating text...")
    text, tps = generate_salad(model, tokenizer)
    print(f"\n[SAMPLE] TPS: {tps:.2f} | Text:\n{text}")
