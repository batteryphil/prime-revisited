import torch
from transformers import AutoTokenizer
import mamba3_titan_mimo
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear

model = TitanMIMO()
lut = build_prime_lut()
for layer in model.layers:
    layer.ssm.in_proj = PrimeLinear(layer.ssm.in_proj, lut)
    layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
for arm in model.mimo_arms:
    arm.ssm.in_proj = PrimeLinear(arm.ssm.in_proj, lut)
    arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)

ckpt_data = torch.load('/data/titan_checkpoints/titan_mimo_live.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt_data['state_dict'], strict=False)
model.cuda().eval()

input_ids_1 = torch.tensor([[10, 20, 30, 40]], device='cuda')
input_ids_2 = torch.tensor([[10, 20, 30, 40, 50]], device='cuda')

with torch.no_grad():
    logits_1, _ = model(input_ids_1, domain_id=0)
    logits_2, _ = model(input_ids_2, domain_id=0)

diff = (logits_1 - logits_2[:, :4, :]).abs().max().item()
print(f"Max Difference: {diff}")
