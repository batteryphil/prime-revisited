import sys, torch
sys.path.append('/home/phil/.gemini/antigravity/scratch/baremetal_test/harmonic_convergence')
from mamba3_prime_native import RealMambaSSM
import mamba3_titan_mimo
mamba3_titan_mimo.Mamba = RealMambaSSM
from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear
model = TitanMIMO()
lut = build_prime_lut()
for layer in model.layers:
    layer.ssm.in_proj = PrimeLinear(layer.ssm.in_proj, lut)
    layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
for arm in model.mimo_arms:
    arm.ssm.in_proj = PrimeLinear(arm.ssm.in_proj, lut)
    arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
ckpt = torch.load('/data/titan_checkpoints/titan_mimo_pilot_4000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()
input_ids = torch.tensor([[100, 200, 300]])
logits, _ = model(input_ids, domain_id=torch.tensor([0]))
print("Logits min/max:", logits.min().item(), logits.max().item())
print("Has NaNs?", torch.isnan(logits).any().item())
