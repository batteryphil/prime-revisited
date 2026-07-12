import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import torch
import torch.nn as nn
from mamba3_titan_mimo import TitanMIMO, CONFIG, PrimeLinear, build_prime_lut

def get_vram(device=0):
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(device) / 1024**2  # MB

def print_memory_delta(name, start_mem):
    current_mem = get_vram()
    delta = current_mem - start_mem
    print(f"[CENSUS] {name:<30} {delta:>8.2f} MB")
    return current_mem

if __name__ == '__main__':
    print("==================================================")
    print("   TITAN MIMO VRAM CENSUS (800M PILOT)")
    print("==================================================")
    
    # 1. Update config to 800M Pilot (4 arms, static routing)
    CONFIG['d_model'] = 1536
    CONFIG['n_layers'] = 40
    CONFIG['mimo_paths'] = 4
    CONFIG['expand'] = 2
    
    # Pre-allocate LUT
    lut = build_prime_lut().cuda()
    
    initial_mem = get_vram()
    mem_ckpt = initial_mem
    
    print(f"[CENSUS] Baseline VRAM:                 {initial_mem:>8.2f} MB")
    
    # 2. Allocate Embeddings
    emb = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model']).cuda()
    mem_ckpt = print_memory_delta("Embeddings (FP32)", mem_ckpt)
    
    # 3. Allocate 1 Backbone Layer (Standard FP32)
    from mamba3_titan_mimo import MambaLayer
    dummy_layer = MambaLayer(CONFIG['d_model']).cuda()
    mem_ckpt = print_memory_delta("1x FP32 MambaLayer", mem_ckpt)
    
    # 4. Wrap with PRIME
    dummy_layer.ssm.in_proj = PrimeLinear(dummy_layer.ssm.in_proj, lut).cuda()
    dummy_layer.ssm.out_proj = PrimeLinear(dummy_layer.ssm.out_proj, lut).cuda()
    mem_ckpt = print_memory_delta("1x PRIME MambaLayer", mem_ckpt)
    
    # 5. Measure Full Model Instantiation
    print("\n--- Full Model Allocation ---")
    start_full = get_vram()
    model = TitanMIMO().cuda()
    for layer in model.layers:
        layer.ssm.in_proj = PrimeLinear(layer.ssm.in_proj, lut).cuda()
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut).cuda()
    for arm in model.mimo_arms:
        arm.ssm.in_proj = PrimeLinear(arm.ssm.in_proj, lut).cuda()
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut).cuda()
    mem_ckpt = print_memory_delta("Full 800M Model (Static)", start_full)
    
    # 6. Optimizer State
    continuous_params = [
        p for n, p in model.named_parameters() 
        if 'base_idx' not in n and 'fine_idx' not in n and 'vote_buffer' not in n
    ]
    optimizer = torch.optim.AdamW(continuous_params, lr=1e-4)
    # Force optimizer state initialization
    for p in continuous_params:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad()
    mem_ckpt = print_memory_delta("AdamW Optimizer State", mem_ckpt)
    
    # 7. Forward Pass (Activations)
    print("\n--- Sequence Dynamics ---")
    seq_len = 256
    batch = 1
    input_ids = torch.randint(0, CONFIG['vocab_size'], (batch, seq_len)).cuda()
    
    start_fwd = get_vram()
    labels = torch.randint(0, CONFIG['vocab_size'], (batch, seq_len)).cuda()
    logits, loss = model(input_ids, labels=labels, domain_id=0)
    mem_ckpt = print_memory_delta("Forward Pass (Activations)", start_fwd)
    
    # 8. Backward Pass (Gradients)
    start_bwd = get_vram()
    loss.backward()
    mem_ckpt = print_memory_delta("Backward Pass (Gradients)", start_bwd)
    
    # Summary
    print("\n==================================================")
    print(f"TOTAL VRAM CONSUMED: {get_vram() - initial_mem:.2f} MB")
    print("==================================================")
