import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import math
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoTokenizer

# Require the fused CUDA kernels for memory efficiency!
from mamba_ssm import Mamba

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'vocab_size':   50304,
    'd_model':      1536,
    'n_layers':     28,  # 28 total backbone layers (~550M Pilot)
    'expand':       2,
    'd_state':      16,
    'mimo_paths':   4,   # 4 explicit domains (Chat, Math, Code, Tool)
    'seq_len':      256,
    'batch_size':   4,
    'grad_accum':   2,
    'lr':           1e-4,
    'total_steps':  100000,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file':   'stats_titan_mimo.json',
    'samples_file': 'samples_titan_mimo.json',
    'log_file':     'training_titan_mimo.log',
}

# ── Logger ────────────────────────────────────────────────────────────────────
class LoggerTee:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, 'a')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
    def isatty(self):
        return False


# ── Prime Harmonic Grid LUT ───────────────────────────────────────────────────
def build_prime_lut(n_points=65536):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    reciprocals = [1.0 / p for p in primes]
    tails = [1.0, 1.5, 2.0]
    anchors = sorted(
        [0.0] + reciprocals + tails + [-r for r in reciprocals] + [-t for t in tails]
    )
    t = torch.linspace(0, 1, n_points, dtype=torch.float32)
    lut = torch.zeros(n_points, dtype=torch.float32)
    n = len(anchors) - 1
    for i in range(n_points):
        pos = t[i].item() * n
        lo = int(pos); hi = min(lo + 1, n); frac = pos - lo
        lut[i] = anchors[lo] * (1 - frac) + anchors[hi] * frac
    return lut

# ── PrimeLinear (Discrete Optimizer) ──────────────────────────────────────────
class PrimeLinear(nn.Module):
    SUPERMAJORITY = 12
    BLOCK_SIZE    = 32
    MAX_STRIDE    = 2
    DECAY_RATE    = 0.90

    def __init__(self, module: nn.Linear, lut: torch.Tensor, name="unnamed"):
        super().__init__()
        self.name = name
        self.in_features  = module.in_features
        self.out_features = module.out_features
        self.lut = lut

        with torch.no_grad():
            w = module.weight.float()
            lut_min, lut_max = lut[0].item(), lut[-1].item()
            span = lut_max - lut_min + 1e-8
            combined = ((w - lut_min) / span * 65535.0).round().clamp(0, 65535).to(torch.int32)
            base = torch.div(combined, 256, rounding_mode='floor').to(torch.uint8)
            fine = (combined % 256).to(torch.uint8)

        n_blocks = (self.out_features * self.in_features) // self.BLOCK_SIZE

        self.register_buffer('base_idx',     base)
        self.register_buffer('fine_idx',     fine)
        self.register_buffer('vote_buffer',  torch.zeros(self.out_features, self.in_features, dtype=torch.int16))
        self.register_buffer('last_dir',     torch.zeros(n_blocks, 1, dtype=torch.int8))
        self.register_buffer('momentum',     torch.zeros(n_blocks, 1, dtype=torch.int8))
        self.register_buffer('rolling_flips', torch.zeros(200, dtype=torch.long))
        self.register_buffer('rolling_idx',  torch.zeros(1, dtype=torch.long))
        self.register_buffer('current_interval_flips', torch.zeros(1, dtype=torch.long))

        self.bias = nn.Parameter(module.bias.data.clone()) if module.bias is not None else None

    # We must provide the .weight property because mamba_ssm accesses it directly!
    @property
    def weight(self):
        combined = self.base_idx.long() * 256 + self.fine_idx.long()
        w = self.lut.to(combined.device)[combined].to(torch.float32)
        if self.training:
            w = w.detach().requires_grad_(True)
            w.register_hook(self._vote_hook)
        return w

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def _vote_hook(self, grad):
        with torch.no_grad():
            # In-place operations to prevent intermediate tensor allocations and massive kernel overhead
            pressure = grad.sign().mul_(10).to(torch.int16)
            self.vote_buffer.add_(pressure).clamp_(-32760, 32760)

    @torch.no_grad()
    def apply_votes(self, lr=1e-4, step=None):
        bs      = self.BLOCK_SIZE
        flat    = self.vote_buffer.view(-1)
        n       = flat.numel()
        aligned = n - (n % bs)
        flat_a  = flat[:aligned]

        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        authorized = (magnitude * (lr / 1e-4) >= self.SUPERMAJORITY)

        combined = (self.base_idx.view(-1)[:aligned].to(torch.int32) * 256 +
                    self.fine_idx.view(-1)[:aligned].to(torch.int32))

        # Return GPU tensors to avoid breaking PyTorch async execution
        vote_pos = (flat_a > 0).float().mean()
        vote_neg = (flat_a < 0).float().mean()

        # Calculate vote entropy even if not authorized
        with torch.no_grad():
            hist = torch.bincount(self.vote_buffer.abs().view(-1), minlength=32761)
            entropy = {
                'zeros': hist[0].item(),
                'low': hist[1:4].sum().item(),
                'med': hist[4:12].sum().item(),
                'high': hist[12:].sum().item()
            }

        # Calculate VPU (Vote Pressure Utilization)
        vpu = (self.vote_buffer.float().abs().mean() / self.SUPERMAJORITY).item()

        if step is not None:
            if step % 50 == 0:
                self.rolling_flips[self.rolling_idx[0]] = self.current_interval_flips[0]
                self.rolling_idx[0] = (self.rolling_idx[0] + 1) % 200
                self.current_interval_flips[0] = 0

        retention_pressure = self.rolling_flips.sum().item() + self.current_interval_flips[0].item()

        if not authorized.any():
            return {
                'flips': 0,
                'vote_pos': vote_pos.item(), 'vote_neg': vote_neg.item(),
                'name': self.name, 'entropy': entropy,
                'vpu': vpu, 'reversals': 0, 'numel': n,
                'retention_pressure': retention_pressure
            }

        step_dir  = torch.sign(blocks)
        block_dir = step_dir.mean(dim=1, keepdim=True).sign().to(torch.int8)
        auth_sq   = authorized.squeeze()
        ld_sq     = self.last_dir.squeeze()
        bd_sq     = block_dir.squeeze()

        reversed_blocks = auth_sq & (bd_sq != ld_sq) & (ld_sq != 0)
        same_dir        = (bd_sq == ld_sq).to(torch.int8)
        new_momentum    = torch.clamp((self.momentum.squeeze() * same_dir + same_dir).to(torch.int8), 0, 8)
        self.momentum   = new_momentum.view(-1, 1)

        cblocks      = combined.view(-1, bs)
        center_dist  = (cblocks - 32768).float().abs().mean(dim=1, keepdim=True)
        base_stride  = torch.clamp((self.MAX_STRIDE * (1.0 - center_dist / 32768.0)).long(), min=1)
        dyn_stride   = torch.clamp(base_stride * (1 + self.momentum.float() / 2.0).to(torch.long), max=self.MAX_STRIDE)
        dyn_stride[reversed_blocks.view(-1, 1)] = 0

        update      = (authorized.float() * step_dir * dyn_stride).to(torch.int32)
        moved       = authorized & ~reversed_blocks.unsqueeze(-1).expand_as(authorized)
        total_flips = moved.sum() * bs

        new_combined = torch.clamp(combined - update.view(-1), 0, 65535)
        self.base_idx.copy_(torch.div(new_combined, 256, rounding_mode='floor').to(torch.uint8).view(self.base_idx.shape))
        self.fine_idx.copy_((new_combined % 256).to(torch.uint8).view(self.fine_idx.shape))
        self.last_dir = block_dir.clone()

        self.vote_buffer.view(-1)[:aligned].view(-1, bs).masked_fill_(authorized, 0)
        
        # Calculate vote entropy before decay
        with torch.no_grad():
            hist = torch.bincount(self.vote_buffer.abs().view(-1), minlength=32761)
            entropy = {
                'zeros': hist[0].item(),
                'low': hist[1:4].sum().item(),
                'med': hist[4:12].sum().item(),
                'high': hist[12:].sum().item()
            }
            
        self.vote_buffer = (self.vote_buffer.float() * self.DECAY_RATE).to(torch.int16)

        reversals = reversed_blocks.sum().item() * bs
        flips = total_flips.item()
        
        if step is not None:
            self.current_interval_flips[0] += flips

        return {
            'flips': flips,
            'vote_pos': vote_pos.item(), 'vote_neg': vote_neg.item(),
            'name': self.name, 'entropy': entropy,
            'vpu': vpu, 'reversals': reversals, 'numel': n,
            'retention_pressure': retention_pressure
        }

class ContinuousVoteOptimizer(torch.optim.Optimizer):
    """
    Applies the PRIME voting logic (high-freq hooks, supermajority gate, zero-on-fire)
    to standard FP32 continuous parameters, dropping AdamW entirely.
    """
    def __init__(self, params, lr=1e-4, supermajority=12, decay=0.90):
        defaults = dict(lr=lr, supermajority=supermajority, decay=decay)
        super().__init__(params, defaults)
        
        # Register hooks and initialize buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['vote_buffer'] = torch.zeros_like(p, dtype=torch.int16)
                
                # Must capture p in default arg to avoid late binding in loop
                def make_hook(p_ref):
                    def hook(grad):
                        with torch.no_grad():
                            state = self.state[p_ref]
                            pressure = grad.sign().mul_(-10).to(torch.int16)
                            state['vote_buffer'].add_(pressure).clamp_(-32760, 32760)
                    return hook
                p.register_hook(make_hook(p))

    @torch.no_grad()
    def step(self, closure=None):
        total_flips = None
        for group in self.param_groups:
            lr = group['lr']
            superm = group['supermajority']
            decay = group['decay']
            
            for p in group['params']:
                state = self.state[p]
                votes = state['vote_buffer']
                
                authorized = votes.abs() >= superm
                
                if authorized.any():
                    # Apply continuous FP32 step scaled by actual gradient
                    if p.grad is not None:
                        update = -p.grad[authorized] * lr
                        p[authorized] += update
                        
                        f = authorized.sum()
                        if total_flips is None:
                            total_flips = f
                        else:
                            total_flips += f
                    
                    votes[authorized] = 0
                
                state['vote_buffer'] = (votes.float() * decay).to(torch.int16)
        
        if total_flips is None:
            return torch.tensor(0, device=self.param_groups[0]['params'][0].device)
        return total_flips

# ── 2B Titan MIMO Architecture ───────────────────────────────────────────────
class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = Mamba(
            d_model=d_model,
            d_state=CONFIG['d_state'],
            d_conv=4,
            expand=CONFIG['expand']
        )
    def forward(self, x):
        return self.ssm(self.norm(x)) + x

class TitanMIMO(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG['d_model']
        v = CONFIG['vocab_size']
        n = CONFIG['n_layers']
        m = CONFIG['mimo_paths']

        self.embedding = nn.Embedding(v, d)
        
        self.layers = nn.ModuleList([MambaLayer(d) for _ in range(n)])
        self._mid = n // 2

        self.mimo_arms = nn.ModuleList([MambaLayer(d) for _ in range(m)])
        self.domain_router = nn.Linear(d, m, bias=True)
        self.router_temp = nn.Parameter(torch.ones(1) * 1.0)
        nn.init.normal_(self.domain_router.weight, std=0.02)
        nn.init.zeros_(self.domain_router.bias)

        self.norm_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, v, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, domain_id=None):
        x = self.embedding(input_ids)

        for layer in self.layers[:self._mid]:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)

        # ── STATIC ROUTING ───────────────────────────────────────────────────
        # Domain IDs: 0=Chat, 1=Math, 2=Code, 3=Tool
        # We build a hard one-hot route weight so gradients strictly target 
        # only the assigned MIMO arm.
        B, L, _ = x.shape
        if domain_id is not None:
            if not isinstance(domain_id, torch.Tensor):
                domain_id = torch.tensor([domain_id]*B, device=x.device)
            elif domain_id.dim() == 0:
                domain_id = domain_id.unsqueeze(0).expand(B)
            one_hot = F.one_hot(domain_id.long(), num_classes=CONFIG['mimo_paths']).float()
            route_weights = one_hot.unsqueeze(1).expand(-1, L, -1)
        else:
            # Fallback uniform routing if no domain provided (e.g. general eval)
            route_weights = torch.ones(B, L, CONFIG['mimo_paths'], device=x.device) / CONFIG['mimo_paths']

        raw_arm_outs = []
        for arm in self.mimo_arms:
            raw_arm_outs.append(arm(x))
        
        stacked = torch.stack(raw_arm_outs, dim=-1)
        collapsed = torch.einsum('b l d m, b l m -> b l d', stacked, route_weights.to(stacked.dtype))
        x = x + collapsed

        for layer in self.layers[self._mid:]:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1)
            )
            # Diversity routing loss removed since we are using static routing.
            # We don't need to load-balance because the dataset loader balances the domains.
        
        return logits, loss

# ── Dataset ──────────────────────────────────────────────────────────────────
def make_dataset(tokenizer):
    import random
    corpus_path = '/data/titan_mega_corpus.jsonl'
    data = []
    domain_map = {'chat': 0, 'math': 1, 'code': 2, 'tool': 3}
    print("[DATA] Loading Mega-Corpus into RAM...")
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj['text'], domain_map.get(obj.get('domain', 'chat'), 0)))
    print(f"[DATA] Loaded {len(data)} high-quality samples.")

    def gen(bsz=CONFIG['batch_size']):
        while True:
            batch_ids = []
            batch_domains = []
            for _ in range(bsz):
                text, domain_id = random.choice(data)
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                tok = tokenizer(text, truncation=True, max_length=CONFIG['seq_len'],
                                padding='max_length', return_tensors='pt')
                batch_ids.append(tok['input_ids'][0])
                batch_domains.append(domain_id)
            
            b_ids = torch.stack(batch_ids)
            yield b_ids, b_ids.clone(), torch.tensor(batch_domains)
    return gen

# ── Thermal Monitor ──────────────────────────────────────────────────────────
def check_thermal_throttle():
    import subprocess
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            encoding='utf-8'
        )
        temp = int(smi.strip())
        if temp >= 85:
            print(f"\n[THERMAL] GPU Temp is {temp}°C (>= 85°C). Throttling for 30s...")
            time.sleep(30)
    except Exception:
        pass

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.stdout = LoggerTee(CONFIG['log_file'])
    sys.stderr = sys.stdout

    print(f"\n{'='*60}")
    print(f"[PRIME] Titan MIMO 650M Agent — PID {os.getpid()}")
    print(f"[{CONFIG['n_layers']} Backbone + {CONFIG['mimo_paths']} Arms]")
    print(f"{'='*60}")

    lut = build_prime_lut()
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INIT] Building 650M MIMO model...")
    model = TitanMIMO()

    wrapped = 0
    for idx, layer in enumerate(model.layers):
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut, name=f"backbone_L{idx}_in_proj")
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut, name=f"backbone_L{idx}_out_proj")
        wrapped += 2
        
    arm_names = ['chat', 'math', 'code', 'tool']
    for idx, arm in enumerate(model.mimo_arms):
        aname = arm_names[idx] if idx < len(arm_names) else str(idx)
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj,  lut, name=f"mimo_arm_{aname}_in_proj")
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut, name=f"mimo_arm_{aname}_out_proj")
        wrapped += 2
    print(f"[INIT] Wrapped {wrapped} linear layers with PrimeDiscrete Optimizer.")

    model = model.to(CONFIG['device'])
    print(f"[INIT] VRAM Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Optimizer for FP32 parameters (embeddings, router, LM head)
    continuous_params = [
        p for n, p in model.named_parameters() 
        if 'base_idx' not in n and 'fine_idx' not in n and 'vote_buffer' not in n
    ]
    total_continuous_params = sum(p.numel() for p in continuous_params)
    optimizer = ContinuousVoteOptimizer(continuous_params, lr=1e-4)

    step = 0
    history = []
    
    # Auto-Resume Logic
    import glob
    checkpoints = glob.glob("/data/titan_checkpoints/*.pt")
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        print(f"[INIT] Resuming from {latest_ckpt}...")
        ckpt_data = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(ckpt_data['state_dict'], strict=False)
        step = ckpt_data['step']
        print(f"[INIT] Successfully restored Step {step}")
        model = model.to(CONFIG['device'])
        
        # We must reconstruct the optimizer after moving to device so the state dict hooks align
        optimizer = ContinuousVoteOptimizer(continuous_params, lr=1e-4)
        total_continuous_params = sum(p.numel() for p in continuous_params)
    data_gen = make_dataset(tokenizer)

    batch_idx  = 0
    total_loss = 0.0
    start_time = time.time()
    accum      = CONFIG['grad_accum']

    torch.cuda.empty_cache()

    print("[TRAIN] Starting 650M Titan Agent training...")
    for input_ids, labels, domain_id in data_gen():
        input_ids = input_ids.to(CONFIG['device'])
        labels    = labels.to(CONFIG['device'])
        domain_id = domain_id.to(CONFIG['device'])

        try:
            _, loss = model(input_ids, labels, domain_id)
        except Exception as e:
            print(f"[WARN] Forward error: {e}")
            continue

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            continue

        check_thermal_throttle()
        (loss / accum).backward()
        total_loss += loss.item()
        batch_idx  += 1

        print('.', end='', flush=True)
        if batch_idx % accum == 0:
            print(f"[{batch_idx}/{accum}] L:{loss.item():.4f}", end=' ', flush=True)

        if batch_idx >= accum:
            step += 1
            print(f"\n[SYNC] Step {step} — applying votes...")
            torch.cuda.empty_cache()

            total_flips = None
            vpos_sum = None
            vneg_sum = None
            count = 0
            
            layer_stats = []
            total_entropy = {'zeros': 0, 'low': 0, 'med': 0, 'high': 0}
            total_reversals = 0
            total_unique = 0

            for m in model.modules():
                if isinstance(m, PrimeLinear):
                    r = m.apply_votes(lr=CONFIG['lr'], step=step)
                    
                    # Track Stats
                    name = r.get('name', f'layer_{count}')
                    flips = r['flips'].item() if hasattr(r['flips'], 'item') else r['flips']
                    vpu = r.get('vpu', 0.0)
                    reversals = r.get('reversals', 0)
                    numel = r.get('numel', 1)
                    retention = r.get('retention_pressure', 0)
                    
                    flip_rate = (flips / max(numel, 1)) * 100.0
                    layer_stats.append({
                        'name': name, 'flips': flips, 'rate': flip_rate,
                        'vpu': vpu, 'reversals': reversals, 'numel': numel,
                        'retention': retention
                    })
                    
                    total_reversals += reversals
                    # If bs=32, unique params flipped = flips / 32
                    total_unique += (flips / 32.0)
                    
                    # Track Entropy
                    e = r.get('entropy', {'zeros':0, 'low':0, 'med':0, 'high':0})
                    total_entropy['zeros'] += e['zeros']
                    total_entropy['low'] += e['low']
                    total_entropy['med'] += e['med']
                    total_entropy['high'] += e['high']
                    
                    if total_flips is None:
                        total_flips = r['flips']
                        vpos_sum = r['vote_pos']
                        vneg_sum = r['vote_neg']
                    else:
                        total_flips += r['flips']
                        vpos_sum += r['vote_pos']
                        vneg_sum += r['vote_neg']
                    count += 1
            
            # Step the continuous parameters via voting
            continuous_flips = optimizer.step()
            c_flips_int = int(continuous_flips.item()) if hasattr(continuous_flips, 'item') else int(continuous_flips)
            
            if total_flips is not None:
                total_flips += continuous_flips
            else:
                total_flips = continuous_flips
                
            if step % 50 == 0 and len(layer_stats) > 0:
                print("\n" + "="*70)
                print("[DIAGNOSTICS] Advanced Learning Allocation")
                total_params = sum(total_entropy.values()) + 1e-8
                print(f"  Vote Entropy | Zeros: {total_entropy['zeros']/total_params*100:.1f}% | Low: {total_entropy['low']/total_params*100:.1f}% | Med: {total_entropy['med']/total_params*100:.1f}% | High: {total_entropy['high']/total_params*100:.1f}%")
                
                t_flips = int(total_flips.item()) if hasattr(total_flips, 'item') else total_flips
                print(f"  Global Flips: {t_flips} (Continuous: {c_flips_int}) | Unique Params Flipped: {int(total_unique)} | Total Reversals: {total_reversals}")
                
                sorted_layers = sorted(layer_stats, key=lambda x: x['rate'], reverse=True)
                print("\n  Top 5 Hyperactive Layers (by %):")
                for s in sorted_layers[:5]:
                    print(f"    {s['name']:>25} | Rate: {s['rate']:>5.2f}% | VPU: {s['vpu']:>4.2f} | Flips: {s['flips']:>8} | Reversals: {s['reversals']}")
                print("\n  Bottom 5 Frozen Layers (by %):")
                for s in sorted_layers[-5:]:
                    print(f"    {s['name']:>25} | Rate: {s['rate']:>5.2f}% | VPU: {s['vpu']:>4.2f} | Flips: {s['flips']:>8} | Reversals: {s['reversals']}")
                print("="*70 + "\n")
                
                # Export to JSON for the dashboard
                telemetry_data = {
                    "step": step,
                    "loss": total_loss / accum,
                    "entropy": {
                        "zeros": total_entropy['zeros']/total_params*100,
                        "low": total_entropy['low']/total_params*100,
                        "med": total_entropy['med']/total_params*100,
                        "high": total_entropy['high']/total_params*100
                    },
                    "global_flips": t_flips,
                    "continuous_flips": c_flips_int,
                    "total_continuous_params": total_continuous_params,
                    "unique_flipped": int(total_unique),
                    "total_reversals": total_reversals,
                    "top_layers": sorted_layers[:5],
                    "bottom_layers": sorted_layers[-5:],
                    "all_layers": sorted_layers
                }
                with open("telemetry_live.json", "w") as f:
                    import json
                    json.dump(telemetry_data, f)

            # Zero gradients for ALL parameters (hook fires during backward, so grads are no longer needed)
            for p in model.parameters():
                p.grad = None

            tps = (CONFIG['seq_len'] * CONFIG['batch_size'] * accum) / max(time.time() - start_time, 1e-6)
            avg_loss = total_loss / accum
            
            # EXACTLY ONE CPU SYNC!
            t_flips = int(total_flips.item()) if hasattr(total_flips, 'item') else (int(total_flips) if total_flips is not None else 0)
            mean_vpos = (vpos_sum.item() / max(count, 1)) if hasattr(vpos_sum, 'item') else ((vpos_sum / max(count, 1)) if vpos_sum is not None else 0.0)
            mean_vneg = (vneg_sum.item() / max(count, 1)) if hasattr(vneg_sum, 'item') else ((vneg_sum / max(count, 1)) if vneg_sum is not None else 0.0)
            
            print(f"[PRIME] Step {step} | Loss: {avg_loss:.4f} | Flips: {t_flips} | TPS: {tps:.1f} | V+:{mean_vpos:.2f} V-:{mean_vneg:.2f}")

            # High-Frequency Live Evaluation (Rolling Checkpoint)
            if step % 100 == 0:
                live_path = "/data/titan_checkpoints/titan_mimo_live.pt"
                torch.save({
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, live_path)
                import sys
                subprocess.Popen([sys.executable, 'async_salad.py', live_path, str(step)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Low-Frequency Archival Checkpointing
            if step % 5000 == 0:
                ckpt_path = f"/data/titan_checkpoints/titan_mimo_pilot_{step}.pt"
                print(f"[SAVE] Archiving Step {step} -> {ckpt_path}")
                torch.save({
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
                
                try:
                    import glob, os
                    ckpt_files = sorted(glob.glob('/data/titan_checkpoints/titan_mimo_pilot_*.pt'), key=os.path.getmtime)
                    if len(ckpt_files) > 5:
                        for old_ckpt in ckpt_files[:-5]:
                            os.remove(old_ckpt)
                except Exception as e:
                    print(f"[CLEANUP ERROR] {e}")

            if step >= CONFIG['total_steps']:
                break

            batch_idx  = 0
            total_loss = 0.0
            start_time = time.time()
            torch.cuda.empty_cache()
