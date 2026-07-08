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
    'total_steps':  10000000,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file':   'stats_titan_mimo.json',
    'samples_file': 'samples_titan_mimo.json',
    'log_file':     'training_titan_mimo.log',
    'corpus_path':  './data/titan_mega_corpus.jsonl',
    'val_path':     './data/titan_val_suite.jsonl',
    'ckpt_dir':     './checkpoints',
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
    BLOCK_SIZE    = 32
    MAX_STRIDE    = 2
    DECAY_RATE    = 0.90

    def __init__(self, module: nn.Linear, lut: torch.Tensor, name="unnamed", **kwargs):
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
        
        import math
        self.target_flip_rate = kwargs.get('target_flip_rate', 0.15)
        
        # PI Controller Buffers (V2)
        self.register_buffer('log_threshold', torch.tensor([math.log(12.0)], dtype=torch.float32))
        self.register_buffer('pi_integral', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('ema_flip_rate', torch.tensor([0.0], dtype=torch.float32))
        
        # Telemetry Buffers
        self.register_buffer('last_grad_sign', torch.zeros(self.out_features, self.in_features, dtype=torch.int8))
        self.register_buffer('sign_agreement_ema', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('delta_log_thresh', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('step_counter', torch.tensor([0], dtype=torch.long))

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
            sign_t = grad.sign().to(torch.int8)
            
            # Compute sign agreement sample every 50 steps
            if self.step_counter[0] % 50 == 0:
                agreement = (sign_t == self.last_grad_sign).float().mean()
                self.sign_agreement_ema[0] = self.sign_agreement_ema[0] * 0.95 + agreement * 0.05
                
            self.last_grad_sign.copy_(sign_t)
            
            pressure = sign_t.mul_(10).to(torch.int16)
            self.vote_buffer.add_(pressure).clamp_(-32760, 32760)

    @torch.no_grad()
    def apply_votes(self, lr=1e-4, step=None):
        bs      = self.BLOCK_SIZE
        flat    = self.vote_buffer.view(-1)
        n       = flat.numel()
        aligned = n - (n % bs)
        flat_a  = flat[:aligned]

        import math
        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        
        self.step_counter[0] += 1
        threshold = torch.exp(self.log_threshold[0])
        authorized = (magnitude * (lr / 1e-4) >= threshold)

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
        vpu = (self.vote_buffer.float().abs().mean() / threshold).item()

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
        
        # V2 Log-Space PI Controller
        with torch.no_grad():
            import math
            current_flip_rate = flips / max(n, 1)
            self.ema_flip_rate[0] = self.ema_flip_rate[0] * 0.999 + current_flip_rate * 0.001
            
            # Run controller every 50 steps, with a 1000 step warmup freeze
            if self.step_counter[0] > 1000 and self.step_counter[0] % 50 == 0:
                target = self.target_flip_rate
                error = (self.ema_flip_rate[0] - target) / max(target, 1e-3)
                
                kp = 0.5
                ki = 0.01
                
                self.pi_integral[0] += error
                adjustment = (kp * error) + (ki * self.pi_integral[0])
                
                new_log_thresh = self.log_threshold[0] + adjustment
                clamped_log = torch.clamp(new_log_thresh, math.log(2.0), math.log(128.0))
                
                self.delta_log_thresh[0] = clamped_log - self.log_threshold[0]
                self.log_threshold[0] = clamped_log
                
                # Anti-windup
                if self.log_threshold[0] >= math.log(128.0) or self.log_threshold[0] <= math.log(2.0):
                    self.pi_integral[0] = 0.0
        
        if step is not None:
            self.current_interval_flips[0] += flips

        return {
            'flips': flips,
            'vote_pos': vote_pos.item(), 'vote_neg': vote_neg.item(),
            'name': self.name, 'entropy': entropy,
            'vpu': vpu, 'reversals': reversals, 'numel': n,
            'retention_pressure': retention_pressure,
            'ema_flip': self.ema_flip_rate[0].item(),
            'sign_agreement': self.sign_agreement_ema[0].item(),
            'threshold': torch.exp(self.log_threshold[0]).item(),
            'delta_thresh': self.delta_log_thresh[0].item()
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
                import math
                state['vote_buffer'] = torch.zeros_like(p, dtype=torch.int16)
                state['log_threshold'] = torch.tensor(math.log(12.0), dtype=torch.float32, device=p.device)
                state['pi_integral'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['ema_flip_rate'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['delta_log_thresh'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['step_counter'] = torch.tensor(0, dtype=torch.long, device=p.device)
                state['target_flip_rate'] = 0.25  # LM head target
                
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
                import math
                state = self.state[p]
                votes = state['vote_buffer']
                state['step_counter'] += 1
                
                thresh = torch.exp(state['log_threshold'])
                authorized = votes.abs() >= thresh
                
                f = 0
                if authorized.any():
                    # Apply continuous FP32 step scaled by actual gradient
                    if p.grad is not None:
                        update = -p.grad[authorized] * lr
                        p[authorized] += update
                        
                        f = authorized.sum().item()
                        if total_flips is None:
                            total_flips = f
                        else:
                            total_flips += f
                    
                    votes[authorized] = 0
                
                state['vote_buffer'] = (votes.float() * decay).to(torch.int16)
                
                # V2 Log-Space PI Controller
                current_flip_rate = f / p.numel()
                state['ema_flip_rate'].mul_(0.999).add_(current_flip_rate * 0.001)
                
                if state['step_counter'].item() > 1000 and state['step_counter'].item() % 50 == 0:
                    target = state['target_flip_rate']
                    error = (state['ema_flip_rate'] - target) / max(target, 1e-3)
                    
                    kp, ki = 0.5, 0.01
                    state['pi_integral'].add_(error)
                    adjustment = (kp * error) + (ki * state['pi_integral'])
                    
                    new_log = state['log_threshold'] + adjustment
                    clamped_log = torch.clamp(new_log, math.log(2.0), math.log(128.0))
                    
                    state['delta_log_thresh'].copy_(clamped_log - state['log_threshold'])
                    state['log_threshold'].copy_(clamped_log)
                    
                    if clamped_log.item() >= math.log(128.0) or clamped_log.item() <= math.log(2.0):
                        state['pi_integral'].fill_(0.0)
        
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
        nn.init.normal_(self.domain_router.weight, std=0.02)
        nn.init.zeros_(self.domain_router.bias)

        self.norm_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, v, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, domain_id=None):
        x = self.embedding(input_ids)

        for layer in self.layers[:self._mid]:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)

        # ── DYNAMIC ROUTING ───────────────────────────────────────────────────
        B, L, _ = x.shape
        # Detach x so routing loss gradients don't destabilize the highly optimized backbone.
        # MUST apply LayerNorm because the mid-network residual variance is huge!
        router_x = F.layer_norm(x.detach(), (x.size(-1),))
        router_logits = self.domain_router(router_x)
        
        if self.training:
            # Use Gumbel-Softmax with hard=True to generate 1-hot routes during training.
            route_weights = F.gumbel_softmax(router_logits, tau=1.0, hard=True, dim=-1)
        else:
            # Inference mode: greedy routing (no Gumbel noise)
            idx = torch.argmax(router_logits, dim=-1)
            route_weights = F.one_hot(idx, num_classes=CONFIG['mimo_paths']).float()
            
        # Detach route_weights so massive downstream residual gradients don't explode the router
        route_weights_detached = route_weights.detach()

        raw_arm_outs = []
        for arm in self.mimo_arms:
            raw_arm_outs.append(arm(x))
        
        stacked = torch.stack(raw_arm_outs, dim=-1)
        collapsed = torch.einsum('b l d m, b l m -> b l d', stacked, route_weights_detached.to(stacked.dtype))
        x = x + collapsed

        for layer in self.layers[self._mid:]:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=0
            )
            
            if domain_id is not None:
                if not isinstance(domain_id, torch.Tensor):
                    domain_id = torch.tensor([domain_id]*B, device=x.device)
                elif domain_id.dim() == 0:
                    domain_id = domain_id.unsqueeze(0).expand(B)
                    
                target = domain_id.unsqueeze(1).expand(-1, L).clone().long()
                
                # Mask out padding tokens (index 0) so the router doesn't learn noise
                target[input_ids == 0] = -100
                
                routing_loss = F.cross_entropy(
                    router_logits.view(-1, CONFIG['mimo_paths']),
                    target.reshape(-1),
                    ignore_index=-100
                )
                loss = loss + (0.5 * routing_loss)
        
        return logits, loss

# ── Dataset ──────────────────────────────────────────────────────────────────
def make_dataset(tokenizer):
    import random
    corpus_path = CONFIG.get('corpus_path', './data/titan_mega_corpus.jsonl')
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

# ── Validation Loop ──────────────────────────────────────────────────────────
def run_validation(model, tokenizer, step, interval_flips):
    import math
    print(f"\n[VALIDATION] Running held-out validation at Step {step}...")
    model.eval()
    domain_map = {'chat': 0, 'math': 1, 'code': 2, 'tool': 3}
    domain_losses = {'chat': [], 'math': [], 'code': [], 'tool': []}
    
    val_path = CONFIG.get('val_path', './data/titan_val_suite.jsonl')
    try:
        with open(val_path) as f:
            for line in f:
                obj = json.loads(line)
                domain = obj.get('domain')
                text = obj.get('text')
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                tok = tokenizer(text, truncation=True, max_length=CONFIG['seq_len'],
                                padding='max_length', return_tensors='pt')
                input_ids = tok['input_ids'].to(CONFIG['device'])
                domain_id = torch.tensor([domain_map.get(domain, 0)]).to(CONFIG['device'])
                
                with torch.no_grad():
                    _, loss = model(input_ids, input_ids.clone(), domain_id)
                    domain_losses[domain].append(loss.item())
                    
        metrics = {"step": step}
        avg_loss_total = 0.0
        count_total = 0
        for d, losses in domain_losses.items():
            if losses:
                avg = sum(losses) / len(losses)
                metrics[f"{d}_loss"] = avg
                metrics[f"{d}_ppl"] = math.exp(avg) if avg < 20 else float('inf')
                avg_loss_total += avg
                count_total += 1
        
        # Calculate plasticity efficiency
        metrics["interval_flips"] = interval_flips
        if count_total > 0:
            metrics["avg_val_loss"] = avg_loss_total / count_total
            # We would compute improvement against the previous val_loss, but we just save it here.
        
        # Compute PI Controller health
        thresholds = []
        for m in model.modules():
            if isinstance(m, PrimeLinear):
                thresholds.append(math.exp(m.log_threshold[0].item()))
        
        if thresholds:
            metrics["avg_threshold"] = sum(thresholds) / len(thresholds)
            metrics["layers_at_ceiling"] = sum(1 for t in thresholds if t >= 127.0)
            metrics["layers_at_floor"] = sum(1 for t in thresholds if t <= 2.1)
        
        with open("val_metrics.jsonl", "a") as vf:
            vf.write(json.dumps(metrics) + "\n")
            
        print(f"[VALIDATION] Metrics saved. Avg Thresh: {metrics.get('avg_threshold', 0):.1f}")
    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")
    finally:
        model.train()

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
    import argparse
    parser = argparse.ArgumentParser(description="Titan MIMO 650M Training Script")
    parser.add_argument('--corpus_path', type=str, default=CONFIG['corpus_path'], help='Path to training mega corpus jsonl')
    parser.add_argument('--val_path', type=str, default=CONFIG['val_path'], help='Path to validation suite jsonl')
    parser.add_argument('--ckpt_dir', type=str, default=CONFIG['ckpt_dir'], help='Directory to load and save checkpoints')
    args = parser.parse_args()
    
    # Update CONFIG with CLI overrides
    CONFIG['corpus_path'] = args.corpus_path
    CONFIG['val_path'] = args.val_path
    CONFIG['ckpt_dir'] = args.ckpt_dir

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
        # Specific target flip rates for hierarchy
        if idx < 7: target = 0.10
        elif idx < 21: target = 0.13
        else: target = 0.16
        
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut, name=f"backbone_L{idx}_in_proj", target_flip_rate=target)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut, name=f"backbone_L{idx}_out_proj", target_flip_rate=target)
        wrapped += 2
        
    arm_names = ['chat', 'math', 'code', 'tool']
    for idx, arm in enumerate(model.mimo_arms):
        aname = arm_names[idx] if idx < len(arm_names) else str(idx)
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj,  lut, name=f"mimo_arm_{aname}_in_proj", target_flip_rate=0.13)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut, name=f"mimo_arm_{aname}_out_proj", target_flip_rate=0.13)
        wrapped += 2
        
    model.domain_router = PrimeLinear(model.domain_router, lut, name="domain_router", target_flip_rate=0.16)
    wrapped += 1
    
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
    interval_flips = 0
    
    # Auto-Resume Logic
    import glob
    os.makedirs(CONFIG['ckpt_dir'], exist_ok=True)
    checkpoints = glob.glob(os.path.join(CONFIG['ckpt_dir'], "*.pt"))
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
        if 'optimizer' in ckpt_data:
            try:
                optimizer.load_state_dict(ckpt_data['optimizer'])
                print("[INIT] Loaded optimizer state successfully.")
            except Exception as e:
                print(f"[INIT] Optimizer state mismatch (likely due to architectural shift). Reinitializing continuous optimizer momentum.")
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
                
            if step % 100 == 0:
                print("\n" + "="*70)
                import math
                print("[DIAGNOSTICS] Homeostatic Controller State:")
                
                # Discrete Layers
                for name, module in model.named_modules():
                    if hasattr(module, 'apply_votes'):
                        thresh = math.exp(module.log_threshold[0].item())
                        rate = module.ema_flip_rate[0].item() * 100
                        target = module.target_flip_rate * 100
                        agmt = module.sign_agreement_ema[0].item() * 100
                        print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | EMA_flip: {rate:4.1f}% (T:{target:.0f}%) | Sign_Agmt: {agmt:4.1f}%")
                        
                # Continuous Layers
                print("\n[DIAGNOSTICS] Continuous Layers (Optimizer):")
                for i, p in enumerate(optimizer.param_groups[0]['params']):
                    state = optimizer.state[p]
                    thresh = math.exp(state['log_threshold'].item())
                    rate = state['ema_flip_rate'].item() * 100
                    target = state['target_flip_rate'] * 100
                    name = f"continuous_param_{i}"
                    for n, param in model.named_parameters():
                        if param is p:
                            name = n
                            break
                    print(f"  {name.rjust(30)} | thresh: {thresh:5.1f} | EMA_flip: {rate:4.1f}% (T:{target:.0f}%)")
                    
                print("="*70 + "\n")
                
                # Setup sorted layers for telemetry export
                sorted_layers = sorted(layer_stats, key=lambda x: x['rate'], reverse=True)
                total_params = sum(total_entropy.values()) + 1e-8
                
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
            interval_flips += t_flips
            mean_vpos = (vpos_sum.item() / max(count, 1)) if hasattr(vpos_sum, 'item') else ((vpos_sum / max(count, 1)) if vpos_sum is not None else 0.0)
            mean_vneg = (vneg_sum.item() / max(count, 1)) if hasattr(vneg_sum, 'item') else ((vneg_sum / max(count, 1)) if vneg_sum is not None else 0.0)
            
            print(f"[PRIME] Step {step} | Loss: {avg_loss:.4f} | Flips: {t_flips} | TPS: {tps:.1f} | V+:{mean_vpos:.2f} V-:{mean_vneg:.2f}")

            # High-Frequency Live Evaluation (Rolling Checkpoint)
            if step % 100 == 0:
                os.makedirs(CONFIG['ckpt_dir'], exist_ok=True)
                live_path = os.path.join(CONFIG['ckpt_dir'], "titan_mimo_live.pt")
                torch.save({
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, live_path)
                # Note: async_salad.py disabled to prevent VRAM competition with the main loop.
                # Use offline_evaluator.py to evaluate capabilities natively on the GPU instead.
            
            # Low-Frequency Archival Checkpointing
            if step % 5000 == 0:
                run_validation(model, tokenizer, step, interval_flips)
                interval_flips = 0  # Reset for next interval
                
                os.makedirs(CONFIG['ckpt_dir'], exist_ok=True)
                ckpt_path = os.path.join(CONFIG['ckpt_dir'], f"titan_mimo_pilot_{step}.pt")
                print(f"[SAVE] Archiving Step {step} -> {ckpt_path}")
                torch.save({
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
                
                try:
                    import glob, os
                    ckpt_files = sorted(glob.glob(os.path.join(CONFIG['ckpt_dir'], 'titan_mimo_pilot_*.pt')), key=os.path.getmtime)
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
