import torch
import mamba3_titan_mimo
import sys
sys.path.append('/home/phil/.gemini/antigravity/scratch/baremetal_test/harmonic_convergence')
from mamba3_prime_native import RealMambaSSM
mamba3_titan_mimo.Mamba = RealMambaSSM
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear
from transformers import AutoTokenizer

model = TitanMIMO()
lut = build_prime_lut()
for layer in model.layers:
    layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
    layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
for arm in model.mimo_arms:
    arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
    arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)

ckpt_data = torch.load('/data/titan_checkpoints/titan_mimo_live.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt_data['state_dict'], strict=False)
model.cpu().eval()

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
input_ids = tokenizer("Q: What is the capital of France?\nA:", return_tensors='pt').input_ids.to('cpu')

with torch.no_grad():
    x = model.embedding(input_ids)
    print("Embedding NaN:", torch.isnan(x).any().item())
    
    # 1. Early Backbone
    for idx, layer in enumerate(model.layers[:model._mid]):
        x = layer(x)
        if torch.isnan(x).any():
            print(f"Layer {idx} NaN!")
            sys.exit(1)
            
    print("Early Backbone NaN:", torch.isnan(x).any().item())
    
    import torch.nn.functional as F
    route_weights = F.softmax(model.domain_router(x.mean(dim=1)), dim=-1)
    raw_arm_outs = []
    for idx, arm in enumerate(model.mimo_arms):
        arm_out = arm(x)
        if torch.isnan(arm_out).any():
            print(f"Arm {idx} NaN!")
            sys.exit(1)
        raw_arm_outs.append(arm_out)
        
    print("MIMO Router NaN:", torch.isnan(torch.stack(raw_arm_outs)).any().item())
    
    stacked = torch.stack(raw_arm_outs, dim=-1)
    collapsed = torch.einsum('b l d m, b l m -> b l d', stacked, route_weights.to(stacked.dtype))
    x = x + collapsed
    
    # 3. Late Backbone
    for idx, layer in enumerate(model.layers[model._mid:]):
        x = layer(x)
        if torch.isnan(x).any():
            print(f"Late Layer {idx} NaN!")
            sys.exit(1)
            
    logits = model.lm_head(model.norm_f(x))
    print("Logits NaN:", torch.isnan(logits).any().item())

