import torch
import os
import glob
import numpy as np

def export_latest_weights():
    # Find latest checkpoint
    ckpts = glob.glob('prime_mamba3_*.pt')
    if not ckpts:
        print("No checkpoints found.")
        return
        
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Loading {latest_ckpt}...")
    
    # Load weights
    sd = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
    state_dict = sd['state_dict']
    
    # Find the LUT
    # It might be in the model state dict or we can just reconstruct it
    lut_key = next((k for k in state_dict.keys() if 'lut' in k), None)
    if lut_key:
        lut = state_dict[lut_key].numpy()
    else:
        # Reconstruct prime LUT if it wasn't saved in state_dict
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        base_states = [p/41.0 for p in primes] + [-p/41.0 for p in primes]
        lut_list = []
        for b in base_states:
            for f in base_states:
                lut_list.append(b + (f * 0.05))
        lut = np.array(lut_list[:65536], dtype=np.float32)
        # Pad to 65536 if necessary
        if len(lut) < 65536:
            pad = np.zeros(65536 - len(lut), dtype=np.float32)
            lut = np.concatenate([lut, pad])

    # Find the first PrimeLinear layer weights
    base_key = next((k for k in state_dict.keys() if 'base_idx' in k), None)
    fine_key = next((k for k in state_dict.keys() if 'fine_idx' in k), None)
    
    if not base_key or not fine_key:
        print("Could not find PrimeLinear indices in state_dict.")
        return
        
    base_idx = state_dict[base_key].numpy().astype(np.uint16)
    fine_idx = state_dict[fine_key].numpy().astype(np.uint16)
    
    # Combine into the 16-bit index expected by the C kernel
    combined = (base_idx * 256 + fine_idx).astype(np.uint16)
    
    print(f"Exporting LUT shape: {lut.shape}")
    print(f"Exporting Weights shape: {combined.shape}")
    
    # Save to raw binary files
    with open('lut.bin', 'wb') as f:
        f.write(lut.tobytes())
        
    with open('weights.bin', 'wb') as f:
        f.write(combined.tobytes())
        
    print("Export complete. Saved lut.bin and weights.bin")
    
    # Write the shapes to a header file for the C kernel
    out_features, in_features = combined.shape
    with open('model_dims.h', 'w') as f:
        f.write(f"#define IN_FEATURES {in_features}\n")
        f.write(f"#define OUT_FEATURES {out_features}\n")

if __name__ == '__main__':
    export_latest_weights()
