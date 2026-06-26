import sys
import time
import json
import torch
import fcntl
from transformers import AutoTokenizer

def generate_salad(model, tokenizer, max_new=30):
    prompt = "Q: What is the capital of France?\nA:"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cpu')
    
    t0 = time.time()
    for _ in range(max_new - 1):
        with torch.no_grad():
            x = model.embedding(input_ids)
            for i, layer in enumerate(model.layers):
                x, _ = layer(x, None)
            
            logits = model.lm_head(model.norm_f(x))
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
        
    # Monkeypatch the fused CUDA Mamba kernel with the pure PyTorch CPU version
    sys.path.append('/home/phil/.gemini/antigravity/scratch/baremetal_test/harmonic_convergence')
    from mamba3_prime_native import RealMambaSSM
    
    # Inject a numerically stable CPU scan that uses a hidden state cache
    def stateful_scan(self, x, h_cache=None):
        xf = x.float()
        dbl = self.x_proj(xf)
        dt_r, B_p, C = dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt_r))
        A = -torch.exp(self.A_log.float())
        
        B, L, D = x.shape
        S = self.d_state
        y = torch.zeros((B, L, D), device=x.device, dtype=x.dtype)
        
        # Initial hidden state or loaded from cache
        if h_cache is None:
            h = torch.zeros((B, D, S), device=x.device, dtype=torch.float32)
        else:
            h = h_cache
            
        for t in range(L):
            dt_t = dt[:, t, :]  # [B, D]
            A_t = torch.exp(dt_t.unsqueeze(-1) * A)  # [B, D, S]
            B_t = B_p[:, t, :]  # [B, S]
            x_t = xf[:, t, :]   # [B, D]
            
            Bu = (dt_t * x_t).unsqueeze(-1) * B_t.unsqueeze(1)
            h = h * A_t + Bu
            
            C_t = C[:, t, :]  # [B, S]
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # [B, D]
            y[:, t, :] = (y_t + x_t * self.D.float()).to(x.dtype)
            
        return y, h
        
    RealMambaSSM._scan = stateful_scan
    
    # Also patch forward to return the cache
    def mamba_forward(self, x, h_cache=None):
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = torch.nn.functional.conv1d(x_in.transpose(1,2), self.conv1d.weight, self.conv1d.bias,
                          padding=self.conv1d.padding[0],
                          groups=self.conv1d.groups)[:, :, :x.shape[1]].transpose(1,2)
        x_conv = torch.nn.functional.silu(x_conv)
        y, h_new = self._scan(x_conv, h_cache)
        return self.out_proj(y * torch.nn.functional.silu(z)), h_new
        
    RealMambaSSM.forward = mamba_forward
    
    import mamba3_titan_mimo
    
    def layer_forward(self, x, h_cache=None):
        y, h_new = self.ssm(self.norm(x), h_cache)
        return y + x, h_new
    mamba3_titan_mimo.MambaLayer.forward = layer_forward
    
    mamba3_titan_mimo.Mamba = RealMambaSSM
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
    model.cpu().eval()
    
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
