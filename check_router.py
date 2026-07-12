import torch
import json
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
tokenizer.pad_token = tokenizer.eos_token

model = TitanMIMO()
lut = build_prime_lut()
for layer in model.layers:
    layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
    layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
for arm in model.mimo_arms:
    arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
    arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
model.domain_router = PrimeLinear(model.domain_router, lut)

ckpt = torch.load("/data/titan_checkpoints/titan_mimo_live.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda().eval()

domain_map = {0: 'chat', 1: 'math', 2: 'code', 3: 'tool'}

with open("/data/titan_val_suite.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        d = obj['domain']
        if d == 'math':
            text = obj['text']
            tok = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
            with torch.no_grad():
                x = model.embedding(tok)
                for layer in model.layers[:model._mid]:
                    x = layer(x)
                import torch.nn.functional as F
                router_x = F.layer_norm(x, (x.size(-1),))
                router_logits = model.domain_router(router_x)
                idx = torch.argmax(router_logits, dim=-1)
                
                common_route = torch.mode(idx[0]).values.item()
                print(f"True Domain: {d.upper():4s} | Router predicted: {domain_map[common_route].upper()}")
                
