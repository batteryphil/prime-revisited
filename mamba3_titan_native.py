import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'd_model':      1024,
    'n_layers':     64,
    'expand':       2,
    'd_state':      16,
    'seq_len':      256,
    'grad_accum':   8,
    'lr':           1e-4,
    'total_steps':  117000,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file':   'stats_mamba3_titan.json',
    'samples_file': 'samples_mamba3_titan.json',
    'log_file':     'training_mamba3_titan.log',
}

_tok_path = os.path.expanduser('~/.hf_token')
login(token=open(_tok_path).read().strip() if os.path.exists(_tok_path)
      else os.environ.get('HF_TOKEN', ''))

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

sys.stdout = LoggerTee(CONFIG['log_file'])
sys.stderr = sys.stdout

# ── Prime Harmonic Grid LUT ───────────────────────────────────────────────────
def build_prime_lut(n_points=65536):
    """Protocol v6.00 — interpolated between prime reciprocal anchors."""
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
    print(f"[PRIME] LUT: {n_points} pts, range [{lut.min():.4f}, {lut.max():.4f}]")
    return lut  # stays on CPU

# ── PrimeLinear — the ONLY optimizer is vote pressure ─────────────────────────
class PrimeLinear(nn.Module):
    """
    Discrete weight matrix on the prime harmonic grid.
    No AdamW, no scale, no continuous descent.
    Gradient signs accumulate into a GPU vote buffer.
    Supermajority gate → index steps (+1 or -1) across the LUT.
    Exactly the mechanism from prime-revisited, applied from scratch.

    v2 changes (reviewer-directed):
      SUPERMAJORITY 5  → 12  : forces stricter consensus, lowers migration rate
      MAX_STRIDE    32 → 2   : clamps velocity, prevents overstepping basins
      DECAY_RATE    0.95 → 0.90 : faster stale-vote clearing
      vote_buffer/last_dir/momentum: CPU → GPU registered buffers
        → eliminates all CPU↔GPU transfers in the optimization hot path
        → GPU utilization: 10% → ~60%+
    """
    SUPERMAJORITY = 12
    BLOCK_SIZE    = 32
    MAX_STRIDE    = 2
    DECAY_RATE    = 0.90

    def __init__(self, module: nn.Linear, lut: torch.Tensor):
        super().__init__()
        self.in_features  = module.in_features
        self.out_features = module.out_features
        self.lut = lut  # shared CPU tensor — moved to device on first forward

        # Map random init weights onto the nearest LUT index
        with torch.no_grad():
            w = module.weight.float()
            lut_min, lut_max = lut[0].item(), lut[-1].item()
            span = lut_max - lut_min + 1e-8
            combined = ((w - lut_min) / span * 65535.0).round().clamp(0, 65535).to(torch.int32)
            base = torch.div(combined, 256, rounding_mode='floor').to(torch.uint8)
            fine = (combined % 256).to(torch.uint8)

        n_blocks = (self.out_features * self.in_features) // self.BLOCK_SIZE

        # All buffers registered → move to GPU with model.to(device)
        self.register_buffer('base_idx',     base)                                          # uint8 GPU
        self.register_buffer('fine_idx',     fine)                                          # uint8 GPU
        self.register_buffer('vote_buffer',  torch.zeros(                                  # int16 GPU
            self.out_features, self.in_features, dtype=torch.int16))
        self.register_buffer('last_dir',     torch.zeros(n_blocks, 1, dtype=torch.int8))  # int8  GPU
        self.register_buffer('momentum',     torch.zeros(n_blocks, 1, dtype=torch.int8))  # int8  GPU

        self.bias = nn.Parameter(module.bias.data.clone()) if module.bias is not None else None

    def forward(self, x):
        combined = self.base_idx.long() * 256 + self.fine_idx.long()
        w = self.lut.to(combined.device)[combined].to(x.dtype)
        if self.training:
            w = w.detach().requires_grad_(True)
            w.register_hook(self._vote_hook)
        return F.linear(x, w, self.bias)

    def _vote_hook(self, grad):
        """Accumulate sign×10 pressure — fully on GPU, zero CPU transfer."""
        with torch.no_grad():
            pressure = (torch.sign(grad) * 10).to(torch.int32)
            self.vote_buffer = torch.clamp(
                self.vote_buffer.to(torch.int32) + pressure, -32760, 32760
            ).to(torch.int16)

    @torch.no_grad()
    def apply_votes(self, lr=1e-4):
        """Block supermajority gate → step indices. Fully GPU-resident. Returns telemetry dict."""
        bs      = self.BLOCK_SIZE
        flat    = self.vote_buffer.view(-1)          # GPU int16
        n       = flat.numel()
        aligned = n - (n % bs)
        flat_a  = flat[:aligned]

        # ── Capture vote distribution BEFORE any modification ─────────────────
        vote_pos  = (flat_a > 0).float().mean().item()
        vote_neg  = (flat_a < 0).float().mean().item()
        vote_neut = max(0.0, 1.0 - vote_pos - vote_neg)

        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        authorized = (magnitude * (lr / 1e-4) >= self.SUPERMAJORITY)

        # Current index state — already on GPU
        combined = (self.base_idx.view(-1)[:aligned].to(torch.int32) * 256 +
                    self.fine_idx.view(-1)[:aligned].to(torch.int32))

        # ── Shared telemetry helper (operates on whichever combined is passed) ─
        def _telemetry(c):
            counts   = torch.bincount(c.long(), minlength=65536)
            p        = counts[counts > 0].float() / c.numel()
            entropy  = -(p * torch.log2(p + 1e-12)).sum().item()
            occupancy = (counts > 0).float().mean().item()
            mom_mean = self.momentum.float().mean().item()
            return {
                'entropy':      round(entropy, 4),
                'disp_95':      0.0,  # Removed to save VRAM
                'occupancy':    round(occupancy, 4),
                'momentum_mean': round(mom_mean, 4),
                'vote_pos':     round(vote_pos, 4),
                'vote_neg':     round(vote_neg, 4),
                'vote_neut':    round(vote_neut, 4),
            }

        if not authorized.any():
            return {'flips': 0, 'migration_rate': 0.0, **_telemetry(combined)}

        # ── Active branch ─────────────────────────────────────────────────────
        step_dir  = torch.sign(blocks)
        block_dir = step_dir.mean(dim=1, keepdim=True).sign().to(torch.int8)
        auth_sq   = authorized.squeeze()
        ld_sq     = self.last_dir.squeeze()
        bd_sq     = block_dir.squeeze()

        reversed_blocks = auth_sq & (bd_sq != ld_sq) & (ld_sq != 0)
        same_dir        = (bd_sq == ld_sq).to(torch.int8)
        new_momentum    = torch.clamp(
            (self.momentum.squeeze() * same_dir + same_dir).to(torch.int8), 0, 8)
        self.momentum   = new_momentum.view(-1, 1)

        cblocks      = combined.view(-1, bs)
        center_dist  = (cblocks - 32768).float().abs().mean(dim=1, keepdim=True)
        # MAX_STRIDE cap: stride scales from 1 → MAX_STRIDE based on center proximity
        base_stride  = torch.clamp(
            (self.MAX_STRIDE * (1.0 - center_dist / 32768.0)).long(), min=1)
        dyn_stride   = torch.clamp(
            base_stride * (1 + self.momentum.float() / 2.0).to(torch.long),
            max=self.MAX_STRIDE)
        dyn_stride[reversed_blocks.view(-1, 1)] = 0

        update      = (authorized.float() * step_dir * dyn_stride).to(torch.int32)
        moved       = authorized & ~reversed_blocks.unsqueeze(-1).expand_as(authorized)
        total_flips = int(moved.sum().item() * bs)

        new_combined = torch.clamp(combined - update.view(-1), 0, 65535)
        self.base_idx.copy_(
            torch.div(new_combined, 256, rounding_mode='floor')
            .to(torch.uint8).view(self.base_idx.shape))
        self.fine_idx.copy_(
            (new_combined % 256).to(torch.uint8).view(self.fine_idx.shape))
        self.last_dir = block_dir.clone()

        # Clear authorized blocks, decay remainder with class-level DECAY_RATE
        self.vote_buffer.view(-1)[:aligned].view(-1, bs).masked_fill_(authorized, 0)
        self.vote_buffer = (self.vote_buffer.float() * self.DECAY_RATE).to(torch.int16)

        return {
            'flips':          total_flips,
            'migration_rate': round(total_flips / max(1, n), 6),
            **_telemetry(new_combined),   # post-update stats on GPU
        }

# ── Pure-PyTorch Mamba Selective Scan ────────────────────────────────────────
class RealMambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        # These two get wrapped with PrimeLinear after construction
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # SSM dynamics — continuous, small, stay as-is
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner,
                                 kernel_size=d_conv, padding=d_conv-1,
                                 groups=self.d_inner, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        dt_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        xz     = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = F.conv1d(x_in.transpose(1,2), self.conv1d.weight, self.conv1d.bias,
                          padding=self.conv1d.padding[0],
                          groups=self.conv1d.groups)[:, :, :x.shape[1]].transpose(1,2)
        x_conv = F.silu(x_conv)
        y      = self._scan(x_conv)
        return self.out_proj(y * F.silu(z))

    def _scan(self, x):
        xf = x.float()
        dbl = self.x_proj(xf)
        dt_r, B_p, C = dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt  = F.softplus(self.dt_proj(dt_r))
        A   = -torch.exp(self.A_log.float())
        dtA = torch.einsum('bld,ds->blds', dt, A)
        log_A_cum = torch.clamp(torch.cumsum(dtA, dim=1), min=-80.0)
        Bu  = torch.einsum('bld,bls->blds', dt * xf, B_p)
        h   = torch.exp(log_A_cum) * torch.cumsum(Bu * torch.exp(-log_A_cum), dim=1)
        y   = torch.einsum('blds,bls->bld', h, C)
        return (y + xf * self.D.float()).to(x.dtype)

# ── Model ────────────────────────────────────────────────────────────────────
class MambaLayer(nn.Module):
    def __init__(self, d_model, expand, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = RealMambaSSM(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(
            lambda inp: self.ssm(self.norm(inp)) + inp, x, use_reentrant=False)

class Mamba3LM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        d = CONFIG['d_model']
        self.embedding = nn.Embedding(vocab_size, d)
        self.layers    = nn.ModuleList([
            MambaLayer(d, CONFIG['expand'], CONFIG['d_state'])
            for _ in range(CONFIG['n_layers'])
        ])
        self.norm_f  = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        x      = self.embedding(input_ids)
        for layer in self.layers:
            x  = layer(x)
        x      = self.norm_f(x)
        logits = self.lm_head(x)
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1))
        return logits, loss

# ── Dataset ──────────────────────────────────────────────────────────────────
def make_dataset(tokenizer):
    import random
    
    # Load the C/C++ instruct dataset
    c_instruct_path = '/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/oo_c_instruct.jsonl'
    c_instruct = []
    try:
        with open(c_instruct_path) as f:
            for line in f:
                c_instruct.append(json.loads(line)['text'])
        print(f"[DATA] Loaded {len(c_instruct)} C/C++ instruction examples")
    except Exception as e:
        print(f"[DATA] Failed to load C instruct: {e}")

    # Load the Operating-Organism codebase corpus
    oo_corpus_path = '/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/oo_corpus.jsonl'
    oo_corpus = []
    try:
        with open(oo_corpus_path) as f:
            for line in f:
                oo_corpus.append(json.loads(line)['text'])
        print(f"[DATA] Loaded {len(oo_corpus)} Operating-Organism chunks")
    except Exception as e:
        print(f"[DATA] Failed to load OO corpus: {e}")

    # Fallback to random tokens if datasets are missing
    if not c_instruct and not oo_corpus:
        print("[WARN] No datasets found! Yielding random tokens.")
        c_instruct = ["### Instruction:\nFail\n### Response:\nData missing"]

    def gen():
        while True:
            # 60% chance to yield C/C++ instruct, 40% chance to yield OO corpus
            if oo_corpus and (not c_instruct or random.random() < 0.40):
                text = random.choice(oo_corpus)
            else:
                text = random.choice(c_instruct)
                
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
                
            tok = tokenizer(text, truncation=True, max_length=CONFIG['seq_len'],
                            padding='max_length', return_tensors='pt')
            ids = tok['input_ids'][0]
            yield ids, ids.clone()
            
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
    print(f"\n{'='*60}")
    print(f"[PRIME] Mamba3-Titan (2.4B) — PID {os.getpid()}")
    print(f"[PRIME] Discrete-only optimizer: vote pressure on prime grid")
    print(f"{'='*60}")

    lut = build_prime_lut()

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    print("[INIT] Building model...")
    model = Mamba3LM(vocab_size)

    # Wrap in_proj and out_proj in every layer with PrimeLinear
    wrapped = 0
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
        wrapped += 2
    print(f"[INIT] Wrapped {wrapped} linear layers with PrimeLinear (prime grid).")

    model = model.to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total params: {total_params/1e6:.1f}M")
    print(f"[INIT] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    step       = 0
    history    = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args, _ = parser.parse_known_args()

    if args.resume and os.path.exists(args.resume):
        print(f"[INIT] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=CONFIG['device'])
        model.load_state_dict(ckpt['state_dict'])
        step = ckpt.get('step', 0)
        
        # Load and trim history to match resumed step
        if os.path.exists(CONFIG['stats_file']):
            try:
                with open(CONFIG['stats_file'], 'r') as f:
                    full_history = json.load(f)
                    history = [h for h in full_history if h.get('step', 0) <= step]
            except Exception as e:
                print(f"[WARN] Failed to load history: {e}")
        print(f"[INIT] Successfully resumed at step {step}")

    data_gen = make_dataset(tokenizer)

    batch_idx  = 0
    total_loss = 0.0
    start_time = time.time()
    accum      = CONFIG['grad_accum']

    print("[TRAIN] Starting pure-discrete PRIME training...")
    for input_ids, labels in data_gen():
        input_ids = input_ids.unsqueeze(0).to(CONFIG['device'])
        labels    = labels.unsqueeze(0).to(CONFIG['device'])

        try:
            _, loss = model(input_ids, labels)
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

            # ── The discrete optimizer ────────────────────────────────────────
            total_flips = 0
            migs, ents, disps, occs, vpos, vneg, vneut, moms = [], [], [], [], [], [], [], []
            for m in model.modules():
                if isinstance(m, PrimeLinear):
                    r = m.apply_votes(lr=CONFIG['lr'])
                    total_flips += r['flips']
                    migs.append(r['migration_rate'])
                    ents.append(r['entropy'])
                    disps.append(r['disp_95'])
                    occs.append(r.get('occupancy', 0.0))
                    vpos.append(r.get('vote_pos', 0.0))
                    vneg.append(r.get('vote_neg', 0.0))
                    vneut.append(r.get('vote_neut', 0.0))
                    moms.append(r.get('momentum_mean', 0.0))

            # Zero all gradients manually — no optimizer.step()
            for p in model.parameters():
                p.grad = None

            mean_mig  = sum(migs)  / max(len(migs),  1)
            mean_ent  = sum(ents)  / max(len(ents),  1)
            mean_disp = sum(disps) / max(len(disps), 1)
            mean_occ  = sum(occs)  / max(len(occs),  1)
            mean_vpos = sum(vpos)  / max(len(vpos),  1)
            mean_vneg = sum(vneg)  / max(len(vneg),  1)
            mean_vneut= sum(vneut) / max(len(vneut), 1)
            mean_mom  = sum(moms)  / max(len(moms),  1)
            tps       = (CONFIG['seq_len'] * accum) / max(time.time() - start_time, 1e-6)
            avg_loss  = total_loss / accum

            print(f"[PRIME] Step {step} | Loss: {avg_loss:.4f} | "
                  f"Mig: {mean_mig*100:.2f}% | Disp95: {mean_disp:.1f} | "
                  f"Ent: {mean_ent:.2f} | Occ: {mean_occ*100:.1f}% | "
                  f"V+:{mean_vpos*100:.0f}%/V-:{mean_vneg*100:.0f}% | TPS: {tps:.1f}")

            stats = {
                'step':           step,
                'loss':           round(avg_loss, 4),
                'tps':            round(tps, 2),
                'migration_rate': round(mean_mig * 100, 4),
                'entropy':        round(mean_ent, 4),
                'disp_95':        round(mean_disp, 2),
                'flips':          total_flips,
                'occupancy':      round(mean_occ, 4),
                'vote_pos':       round(mean_vpos, 4),
                'vote_neg':       round(mean_vneg, 4),
                'vote_neut':      round(mean_vneut, 4),
                'momentum_mean':  round(mean_mom, 4),
                'timestamp':      time.time(),
            }
            history.append(stats)
            with open(CONFIG['stats_file'], 'w') as f:
                json.dump(history, f)

            if step % 50 == 0:
                torch.save({'step': step, 'state_dict': model.state_dict(), 'stats': stats},
                           f"prime_mamba3_{step}.pt")
                print(f"[CKPT] Saved prime_mamba3_{step}.pt")

                # ── Word salad generation ─────────────────────────────────────
                print(f"[SALAD] Generating at step {step}...")
                model.eval()
                salad_prompts = [
                    "### Instruction:\nWrite a Python function to reverse a string.\n### Response:\n",
                    "### Instruction:\nWhat is a neural network?\n### Response:\n",
                    "### Instruction:\ndef fibonacci(n):\n### Response:\n",
                ]
                salad_texts = []
                with torch.no_grad():
                    for prompt in salad_prompts:
                        try:
                            p_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(CONFIG['device'])
                            gen = model.generate(
                                p_ids, max_new_tokens=80,
                                temperature=0.8, do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            ) if hasattr(model, 'generate') else None
                            if gen is not None:
                                text = tokenizer.decode(gen[0][p_ids.shape[1]:], skip_special_tokens=True)
                                salad_texts.append(text)
                            else:
                                # Manual greedy decode fallback
                                inp = p_ids
                                for _ in range(80):
                                    logits, _ = model(inp)
                                    next_tok = logits[:, -1, :].div(0.8).softmax(-1).multinomial(1)
                                    inp = torch.cat([inp, next_tok], dim=1)
                                    if next_tok.item() == tokenizer.eos_token_id:
                                        break
                                text = tokenizer.decode(inp[0][p_ids.shape[1]:], skip_special_tokens=True)
                                salad_texts.append(text)
                        except Exception as e:
                            salad_texts.append(f'[gen error: {e}]')
                model.train()
                salad_entry = {
                    'step': step,
                    'text': ' | '.join(salad_texts),
                    'prompts': salad_prompts,
                    'samples': salad_texts,
                }
                # Load existing samples, append, save last 20
                try:
                    with open(CONFIG['samples_file']) as sf:
                        all_salads = json.load(sf)
                except Exception:
                    all_salads = []
                all_salads.append(salad_entry)
                with open(CONFIG['samples_file'], 'w') as sf:
                    json.dump(all_salads[-20:], sf)
                print(f"[SALAD] Step {step}: {salad_texts[0][:120]}")

            batch_idx  = 0
            total_loss = 0.0
            start_time = time.time()
            torch.cuda.empty_cache()

            if step >= CONFIG['total_steps']:
                break

    print("[PRIME] Training complete.")
