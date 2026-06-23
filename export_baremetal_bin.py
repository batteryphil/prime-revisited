import struct
import torch
import os
import glob
from mamba3_prime_native import build_prime_lut

def main():
    # Find latest checkpoint
    ckpts = sorted(glob.glob('prime_mamba3_*.pt'),
                   key=lambda f: int(f.split('_')[-1].replace('.pt', '')))
    ckpt_path = ckpts[-1]
    print(f"Loading {ckpt_path}...")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    d_model = 1024
    n_layers = 28
    vocab_size = sd['embedding.weight'].shape[0]
    lut_size = 65536
    
    lut = build_prime_lut()

    out_path = ckpt_path.replace('.pt', '.bin')
    print(f"Exporting to {out_path}...")

    with open(out_path, 'wb') as f:
        # 1. Header (256 bytes)
        # magic: 0x5052494D ('PRIM')
        magic = 0x5052494D
        header = struct.pack('iiiii', magic, d_model, n_layers, vocab_size, lut_size)
        header += b'\x00' * (256 - len(header))
        f.write(header)

        # 2. LUT
        f.write(lut.numpy().astype('float32').tobytes())

        # 3. Embeddings
        f.write(sd['embedding.weight'].numpy().astype('float32').tobytes())

        # 4. Layers
        for i in range(n_layers):
            prefix = f'layers.{i}.'
            
            # Norm
            f.write(sd[f'{prefix}norm.weight'].numpy().astype('float32').tobytes())
            f.write(sd[f'{prefix}norm.bias'].numpy().astype('float32').tobytes())
            
            # SSM constants
            f.write(sd[f'{prefix}ssm.A_log'].numpy().astype('float32').tobytes())
            f.write(sd[f'{prefix}ssm.D'].numpy().astype('float32').tobytes())
            
            # in_proj_idx (uint16_t)
            base = sd[f'{prefix}ssm.in_proj.base_idx'].to(torch.int32)
            fine = sd[f'{prefix}ssm.in_proj.fine_idx'].to(torch.int32)
            combined = (base * 256 + fine).to(torch.int16)
            f.write(combined.numpy().tobytes())
            
            # conv1d
            f.write(sd[f'{prefix}ssm.conv1d.weight'].numpy().astype('float32').tobytes())
            f.write(sd[f'{prefix}ssm.conv1d.bias'].numpy().astype('float32').tobytes())
            
            # x_proj
            f.write(sd[f'{prefix}ssm.x_proj.weight'].numpy().astype('float32').tobytes())
            
            # dt_proj
            f.write(sd[f'{prefix}ssm.dt_proj.weight'].numpy().astype('float32').tobytes())
            f.write(sd[f'{prefix}ssm.dt_proj.bias'].numpy().astype('float32').tobytes())
            
            # out_proj_idx (uint16_t)
            base_out = sd[f'{prefix}ssm.out_proj.base_idx'].to(torch.int32)
            fine_out = sd[f'{prefix}ssm.out_proj.fine_idx'].to(torch.int32)
            combined_out = (base_out * 256 + fine_out).to(torch.int16)
            f.write(combined_out.numpy().tobytes())

        # 5. Final Norm & LM Head
        f.write(sd['norm_f.weight'].numpy().astype('float32').tobytes())
        f.write(sd['norm_f.bias'].numpy().astype('float32').tobytes())
        f.write(sd['lm_head.weight'].numpy().astype('float32').tobytes())

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Export complete. Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
