import torch
import math
import sys
from transformers import AutoTokenizer
sys.path.append("/home/phil/.gemini/antigravity/scratch/analysis_project/titan_venv/lib/python3.12/site-packages")
from mamba3_prime_native import Mamba3LM, build_prime_lut
t = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
if t.pad_token is None: t.pad_token = t.eos_token
model = Mamba3LM(len(t), build_prime_lut(device='cpu'))
ff_count = 0
state_count = 0
other_count = 0

def count_module(m, name_prefix):
    global ff_count, state_count, other_count
    for n, p in m.named_parameters(recurse=False):
        c = p.numel()
        if 'prime_ff' in name_prefix or 'out_proj' in name_prefix: ff_count += c
        elif 'continuous_dyn' in name_prefix or 'bias' in n or 'D' in n: state_count += c
        else: other_count += c
    for n, b in m.named_buffers(recurse=False):
        if 'idx' in n: # base_idx, fine_idx
            c = b.numel()
            if 'prime_ff' in name_prefix or 'out_proj' in name_prefix: ff_count += c
            else: other_count += c

for name, module in model.named_modules():
    count_module(module, name)

print(f"Total Model Elements: {(ff_count + state_count + other_count)/1e6:.2f}M")
print(f"Feed-Forward Elements (z, x, out): {ff_count/1e6:.2f}M")
print(f"State/Routing Elements (B, C, dt, trap): {state_count/1e6:.2f}M")
print(f"Other Elements (Embeds, Heads, Norms): {other_count/1e6:.2f}M")
