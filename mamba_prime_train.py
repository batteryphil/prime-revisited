"""
Mamba-130m PRIME Training — Trinity Approach + Voting System
============================================================
Uses state-spaces/mamba-130m-hf as base.
Trinity pattern: swap nn.Linear -> PrimeLinear (STE forward, CPU vote buffer).
Voting: accumulate gradient sign pressure, apply discrete weight flips each step.
Answer-masked SFT on Python instructions (same dataset as Qwen run).

Key improvements over Qwen attempt:
  - All vote_buffers born on CPU (never touch GPU VRAM)
  - No multi-process log pollution (clean PID tracking)
  - Proper empty_cache() before vote application
  - Single unified training loop
  - Small model (130M) = fits easily in 12GB
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'model_name': 'state-spaces/mamba-130m-hf',
    'seq_len': 512,
    'grad_accum': 32,
    'lr': 1e-4,
    'total_steps': 30,
    'noise_every': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file': 'stats_prime.json',
    'samples_file': 'samples_prime.json',
    'checkpoint_prefix': 'prime_step',
    'log_file': 'training_prime.log',
    # [DIAGNOSTIC] Freeze scale for first N steps to test gradient-sink hypothesis.
    # If vote entropy / migration rate jump when scale is frozen, scale is the sink.
    'freeze_scale_steps': 0,
    # [BASELINE] Set True to skip index flipping (train scale only) to compare PRIME vs Quantized.
    'baseline_mode': False,
}

# ── Logging tee ───────────────────────────────────────────────────────────────
class LoggerTee:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, 'a')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return False

sys.stdout = LoggerTee(CONFIG['log_file'])
sys.stderr = sys.stdout

print(f"\n{'='*60}")
print(f"[PRIME] Mamba-130m PRIME Training — PID {os.getpid()}")
print(f"[PRIME] Device: {CONFIG['device']}")
print(f"{'='*60}")

# ── Prime Harmonic Grid ───────────────────────────────────────────────────────
def build_prime_lut(n_points=65536, device='cpu'):
    """16-bit Prime Harmonic Manifold — same as Qwen run, Protocol v6.00"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    reciprocals = [1.0 / p for p in primes]
    tails = [1.0, 1.5, 2.0]
    anchors = sorted(
        [0.0] + reciprocals + tails + [-r for r in reciprocals] + [-t for t in tails]
    )
    anchor_t = torch.tensor(anchors, dtype=torch.float32)
    t = torch.linspace(0, 1, n_points, dtype=torch.float32)
    # Interpolate between anchors
    lut = torch.zeros(n_points, dtype=torch.float32)
    n = len(anchors) - 1
    for i in range(n_points):
        pos = t[i].item() * n
        lo = int(pos)
        hi = min(lo + 1, n)
        frac = pos - lo
        lut[i] = anchors[lo] * (1 - frac) + anchors[hi] * frac
    print(f"[PRIME] LUT built: {n_points} points, range [{lut.min():.4f}, {lut.max():.4f}]")
    return lut.to(device)

# ── Prime Linear Layer (Trinity-style in-place wrap + Voting) ─────────────────
class PrimeLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear.
    Forward: dequantize from 16-bit index -> float weight, apply linear.
    Backward: gradient signs voted into CPU vote buffer.
    Vote apply: discrete index steps on CPU each training step.
    """
    SUPERMAJORITY = 5    # Lowered from 20 — was causing vote stagnation (neut ~99%)
    BLOCK_SIZE = 32

    def __init__(self, original: nn.Linear, lut: torch.Tensor):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.has_bias = original.bias is not None
        self.lut = lut  # shared reference, stays on CPU

        # Scale per output row (learnable, on GPU)
        self.scale = nn.Parameter(torch.ones(self.out_features, 1))

        # Quantize original FP16 weights to 16-bit LUT index — fast direct mapping.
        # LUT is linearly interpolated in [-2, 2], so nearest index = clamp(round((w+2)/4 * 65535), 0, 65535)
        with torch.no_grad():
            w = original.weight.float()  # (out, in)
            lut_min, lut_max = lut[0].item(), lut[-1].item()
            span = lut_max - lut_min + 1e-8
            combined = ((w - lut_min) / span * 65535.0).round().clamp(0, 65535).to(torch.int32)
            combined = combined.view(self.out_features, self.in_features)
            base = torch.div(combined, 256, rounding_mode='floor').to(torch.uint8)
            fine = (combined % 256).to(torch.uint8)

        # Indices stay on GPU for fast forward; vote buffer born on CPU
        self.register_buffer('base_idx', base)
        self.register_buffer('fine_idx', fine)
        self.vote_buffer = torch.zeros(self.out_features, self.in_features,
                                       dtype=torch.int16)  # CPU only

        # ── Trajectory tracking (CPU) ─────────────────────────────────────────
        # init_combined: starting LUT index, for net displacement measurement
        self.register_buffer('init_combined',
            combined.clone().to(torch.int32))  # stays CPU via init
        # last_dir: direction of last block move, for reversal rate
        n_blocks = (self.out_features * self.in_features) // self.BLOCK_SIZE
        self.last_dir = torch.zeros(n_blocks, 1, dtype=torch.int8)  # CPU
        # momentum: consecutive steps in same direction per block (caps at 8)
        self.momentum = torch.zeros(n_blocks, 1, dtype=torch.int8)  # CPU

        if self.has_bias:
            self.bias = nn.Parameter(original.bias.data.clone())
        else:
            self.bias = None

    @property
    def weight(self):
        """Expose dequantized weight for HF model internals that access .weight directly."""
        return self._get_weights().detach()

    def _get_weights(self):
        """Dequantize: (out, in) float tensor on same device as scale."""
        combined = self.base_idx.long() * 256 + self.fine_idx.long()  # GPU
        lut_gpu = self.lut.to(combined.device)
        return lut_gpu[combined] * self.scale  # (out, in)

    def forward(self, x):
        w = self._get_weights()
        if self.training:
            # Force autograd tracking so hook fires even when scale is frozen.
            w = w + torch.zeros_like(w, requires_grad=True)
            w.register_hook(self._vote_hook)
        out = F.linear(x, w, self.bias)
        return out

    def _vote_hook(self, grad):
        """Accumulate gradient pressure into CPU vote buffer.

        Uses sign-based pressure (not magnitude-normalized) per the reviewing AI:
        grad/grad_max over-normalizes — one outlier dominates grad_max and most
        weights see pressure=0 after int cast. sign(grad)*10 gives every weight
        exactly ±10 per batch, making threshold crossings predictable and consistent.
        """
        with torch.no_grad():
            # sign-based: ±10 per batch per weight, uniform signal strength
            pressure = (torch.sign(grad).cpu() * 10).to(torch.int32)
            new = torch.clamp(
                self.vote_buffer.to(torch.int32) + pressure, -32760, 32760
            )
            self.vote_buffer = new.to(torch.int16)

    @torch.no_grad()
    def apply_votes(self, lr=1e-4):
        """Apply accumulated votes. Returns health + trajectory metrics."""
        block_size = self.BLOCK_SIZE
        flat = self.vote_buffer.view(-1)
        n = flat.numel()
        aligned = n - (n % block_size)
        flat_v = flat[:aligned]
        num_blocks = aligned // block_size
        blocks = flat_v.float().view(num_blocks, block_size)

        # ── Vote entropy & magnitude ───────────────────────────────────────────
        pos_frac      = (flat > 0).float().mean().item()
        neg_frac      = (flat < 0).float().mean().item()
        neut_frac     = 1.0 - pos_frac - neg_frac
        vote_abs_mean = flat.float().abs().mean().item()

        magnitude = blocks.abs().mean(dim=1, keepdim=True)
        pressure_scale = lr / 1e-4
        authorized = (magnitude * pressure_scale >= self.SUPERMAJORITY)

        if not authorized.any() or CONFIG.get('baseline_mode', False):
            base_cpu = self.base_idx.cpu()
            fine_cpu = self.fine_idx.cpu()
            combined = (base_cpu.view(-1)[:aligned].to(torch.int32) * 256 + fine_cpu.view(-1)[:aligned].to(torch.int32))
            sample_idx = combined[::max(1, len(combined)//4096)]
            
            sample_w = self.lut[sample_idx.to(torch.long)]
            w_abs_mean = sample_w.abs().mean().item()
            w_std = sample_w.std().item()
            near_zero = (sample_w.abs() < 1e-3).float().mean().item()
            idx_std = sample_idx.float().std().item()

            occupied_bins = torch.zeros(65536, dtype=torch.bool, device=combined.device)
            occupied_bins[combined.long()] = True
            
            counts = torch.bincount(combined.long(), minlength=65536)
            p = counts[counts > 0].float() / combined.numel()
            entropy = -(p * torch.log2(p)).sum().item()

            return {'flips': 0, 'migration_rate': 0.0, 'occupied_bins': occupied_bins,
                    'vote_pos': pos_frac, 'vote_neg': neg_frac,
                    'vote_neut': neut_frac, 'vote_abs_mean': vote_abs_mean,
                    'reversal_rate': 0.0, 'net_displacement': 0.0,
                    'w_abs_mean': round(w_abs_mean, 4), 'w_std': round(w_std, 4),
                    'near_zero': round(near_zero, 4), 'idx_std': round(idx_std, 2),
                    'entropy': round(entropy, 4), 'disp_95': 0.0, 'momentum_mean': 0.0}

        step_dir = torch.sign(blocks)  # (num_blocks, block_size)
        # Dominant direction per block: +1 or -1
        block_dir = step_dir.mean(dim=1, keepdim=True).sign().to(torch.int8)

        # ── Reversal rate ──────────────────────────────────────────────────────
        # Fraction of authorized blocks that flipped vs last step
        auth_sq = authorized.squeeze()
        ld_sq   = self.last_dir.squeeze()
        bd_sq   = block_dir.squeeze()
        reversed_blocks = auth_sq & (bd_sq != ld_sq) & (ld_sq != 0)
        reversal_rate = reversed_blocks.float().mean().item() if auth_sq.any() else 0.0

        # ── Momentum-gated stride ──────────────────────────────────────────────
        # Same direction as last step → momentum += 1 (cap 8) → bigger stride
        # Reversed direction → momentum = 0 → no move this step (penalise oscillation)
        same_dir    = (bd_sq == ld_sq).to(torch.int8)
        new_momentum = torch.clamp(
            (self.momentum.squeeze() * same_dir + same_dir).to(torch.int8), 0, 8)
        self.momentum = new_momentum.view(-1, 1)

        base_cpu = self.base_idx.cpu()
        fine_cpu = self.fine_idx.cpu()
        combined = (base_cpu.view(-1)[:aligned].to(torch.int32) * 256 +
                    fine_cpu.view(-1)[:aligned].to(torch.int32))
        combined_blocks = combined.view(num_blocks, block_size)

        center_dist = torch.abs(combined_blocks - 32768).float().mean(dim=1, keepdim=True)
        stride_decay = center_dist / 32768.0
        base_stride = torch.clamp((8.0 * (1.0 - stride_decay)).long(), min=1)

        # Momentum multiplier: persistent direction gets up to 4x stride
        momentum_mult = (1 + self.momentum.float() / 2.0).to(torch.long)
        dynamic_stride = base_stride * momentum_mult
        # Zero stride for reversing blocks — no reward for oscillation
        dynamic_stride[reversed_blocks.view(-1, 1)] = 0

        update = (authorized.float() * step_dir * dynamic_stride).to(torch.int32)
        # Only count non-zeroed blocks as flips
        moved = authorized & ~reversed_blocks.unsqueeze(-1).expand_as(authorized)
        total_flips = int(moved.sum().item() * block_size)

        new_combined = torch.clamp(combined - update.view(-1), 0, 65535)
        new_base = torch.div(new_combined, 256, rounding_mode='floor').to(torch.uint8)
        new_fine  = (new_combined % 256).to(torch.uint8)

        self.base_idx.copy_(new_base.view(self.base_idx.shape).to(self.base_idx.device))
        self.fine_idx.copy_(new_fine.view(self.fine_idx.shape).to(self.fine_idx.device))

        # Update last_dir for next step
        self.last_dir = block_dir.clone()

        # Decay vote buffer
        self.vote_buffer.view(-1)[:aligned].view(num_blocks, block_size).masked_fill_(authorized, 0)
        self.vote_buffer = (self.vote_buffer.float() * 0.95).to(torch.int16)

        # ── Occupancy & Weight Distribution ────────────────────────────────────
        occupied_bins = torch.zeros(65536, dtype=torch.bool, device=new_combined.device)
        occupied_bins[new_combined.long()] = True
        
        counts = torch.bincount(new_combined.long(), minlength=65536)
        p = counts[counts > 0].float() / new_combined.numel()
        entropy = -(p * torch.log2(p)).sum().item()

        sample_idx = new_combined[::max(1, len(new_combined)//4096)]
        sample_w   = self.lut[sample_idx.to(torch.long)]
        w_abs_mean = sample_w.abs().mean().item()
        w_std      = sample_w.std().item()
        near_zero  = (sample_w.abs() < 1e-3).float().mean().item()
        idx_std    = sample_idx.float().std().item()

        # ── Net displacement from initial state ────────────────────────────────
        init_flat   = self.init_combined.cpu().view(-1)[:aligned]
        diff        = (new_combined - init_flat).float().abs()
        net_disp    = diff.mean().item()
        
        sample_diff = diff[::max(1, len(diff)//100000)]
        disp_95     = torch.quantile(sample_diff, 0.95).item()
        
        momentum_mean = self.momentum.float().mean().item()

        migration_rate = total_flips / max(1, n)

        return {
            'flips':           total_flips,
            'migration_rate':  round(migration_rate, 6),
            'occupied_bins':   occupied_bins,
            'vote_pos':        round(pos_frac, 4),
            'vote_neg':        round(neg_frac, 4),
            'vote_neut':       round(neut_frac, 4),
            'vote_abs_mean':   round(vote_abs_mean, 4),
            'reversal_rate':   round(reversal_rate, 4),
            'net_displacement': round(net_disp, 2),
            'w_abs_mean':      round(w_abs_mean, 4),
            'w_std':           round(w_std, 4),
            'near_zero':       round(near_zero, 4),
            'idx_std':         round(idx_std, 2),
            'entropy':         round(entropy, 4),
            'disp_95':         round(disp_95, 2),
            'momentum_mean':   round(momentum_mean, 4),
        }


    @torch.no_grad()
    def inject_noise(self, jitter=1):
        """Trinity: thermal noise to prevent Phase 4 memorization collapse."""
        noise = torch.randint(-jitter, jitter + 1, self.vote_buffer.shape, dtype=torch.int16)
        self.vote_buffer.add_(noise)


# ── Model setup ───────────────────────────────────────────────────────────────
def load_and_wrap_model(lut):
    print(f"[PRIME] Loading {CONFIG['model_name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float32,
    )

    # Swap all nn.Linear layers with PrimeLinear (Trinity pattern)
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parts = name.rsplit('.', 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            attr = parts[-1]
            prime = PrimeLinear(module, lut)
            setattr(parent, attr, prime)
            replaced += 1

    print(f"[PRIME] Wrapped {replaced} linear layers with PrimeLinear.")
    model = model.to(CONFIG['device'])
    torch.cuda.empty_cache()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[PRIME] VRAM used after load: {vram:.2f} GB")
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_dataloader(tokenizer):
    dataset = load_dataset(
        'iamtarun/python_code_instructions_18k_alpaca',
        split='train', streaming=True
    )

    def collate(batch):
        texts = []
        for ex in batch:
            inst = ex.get('instruction', '')
            out = ex.get('output', ex.get('response', ''))
            texts.append(f"### Instruction:\n{inst}\n### Response:\n{out}{tokenizer.eos_token}")

        enc = tokenizer(texts, return_tensors='pt', padding=True,
                        truncation=True, max_length=CONFIG['seq_len'])
        ids = enc['input_ids']

        labels = ids.clone()
        # Answer-mask: find "### Response:\n" and mask everything before it
        resp_ids = tokenizer.encode('### Response:\n', add_special_tokens=False)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1] - len(resp_ids)):
                if ids[i, j:j+len(resp_ids)].tolist() == resp_ids:
                    labels[i, :j+len(resp_ids)] = -100
                    break
            else:
                labels[i, :] = -100  # no response found, skip whole sample

        return ids, labels

    return dataset, collate


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    lut = build_prime_lut(n_points=65536, device='cpu')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = load_and_wrap_model(lut)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG['lr'], weight_decay=0.01
    )

    # ── Scale freeze diagnostic ──────────────────────────────────────────────
    # Freeze scale params for the first N steps to test if scale is a gradient
    # sink. If vote entropy / migration rate jump during freeze, hypothesis confirmed.
    freeze_steps = CONFIG.get('freeze_scale_steps', 0)
    scale_params = [m.scale for m in model.modules() if isinstance(m, PrimeLinear)]
    if freeze_steps > 0:
        for p in scale_params:
            p.requires_grad = False
        print(f"[DIAGNOSTIC] Scale frozen for first {freeze_steps} steps. Watching vote activity...")

    dataset, collate = make_dataloader(tokenizer)

    print(f"\n[PRIME] Starting training — {CONFIG['total_steps']} steps, "
          f"grad_accum={CONFIG['grad_accum']}, seq={CONFIG['seq_len']}")

    step = 0
    batch_idx = 0
    total_loss = 0.0
    start_time = time.time()
    history = []
    samples = []

    data_iter = iter(dataset)
    accum_steps = CONFIG['grad_accum']

    try:
        old_stats = json.load(open(CONFIG['stats_file'])) if os.path.exists(CONFIG['stats_file']) else []
    except Exception:
        old_stats = []

    optimizer.zero_grad()

    while step < CONFIG['total_steps']:
        # Gather a micro-batch
        raw_batch = []
        for _ in range(1):
            try:
                raw_batch.append(next(data_iter))
            except StopIteration:
                data_iter = iter(dataset)
                raw_batch.append(next(data_iter))

        try:
            input_ids, labels = collate(raw_batch)
        except Exception:
            continue

        input_ids = input_ids.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        loss_val = loss.item() * accum_steps
        total_loss += loss_val

        batch_idx += 1
        print(f".", end='', flush=True)
        if batch_idx % 10 == 0:
            tps = (CONFIG['seq_len'] * batch_idx) / (time.time() - start_time + 1e-6)
            print(f"[{batch_idx}/{accum_steps}] L:{loss_val:.4f} TPS:{tps:.1f}", end=' ', flush=True)

        if batch_idx >= accum_steps:
            step += 1
            print(f"\n[SYNC] Step {step} finalizing...")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # ── Vote application (all on CPU) ──
            torch.cuda.empty_cache()
            total_flips = 0
            mig_rates = []
            vote_pos_list, vote_neg_list, vote_neut_list, vote_abs_list = [], [], [], []
            reversal_list, net_disp_list = [], []
            w_abs_list, w_std_list, near_zero_list, idx_std_list = [], [], [], []
            entropy_list, disp_95_list, momentum_list = [], [], []
            global_bins = torch.zeros(65536, dtype=torch.bool)
            
            for module in model.modules():
                if isinstance(module, PrimeLinear):
                    r = module.apply_votes(lr=CONFIG['lr'])
                    if isinstance(r, dict):
                        total_flips        += r['flips']
                        mig_rates.append(r['migration_rate'])
                        global_bins |= r['occupied_bins'].cpu()
                        vote_pos_list.append(r['vote_pos'])
                        vote_neg_list.append(r['vote_neg'])
                        vote_neut_list.append(r['vote_neut'])
                        vote_abs_list.append(r.get('vote_abs_mean', 0))
                        reversal_list.append(r.get('reversal_rate', 0))
                        net_disp_list.append(r.get('net_displacement', 0))
                        w_abs_list.append(r.get('w_abs_mean', 0))
                        w_std_list.append(r.get('w_std', 0))
                        near_zero_list.append(r.get('near_zero', 0))
                        idx_std_list.append(r.get('idx_std', 0))
                        entropy_list.append(r.get('entropy', 0))
                        disp_95_list.append(r.get('disp_95', 0))
                        momentum_list.append(r.get('momentum_mean', 0))
                    else:
                        total_flips += r

            migration_rate  = round(sum(mig_rates) / max(len(mig_rates), 1), 6)
            occupancy       = round(global_bins.float().mean().item(), 4)
            vote_pos        = round(sum(vote_pos_list) / max(len(vote_pos_list), 1), 4)
            vote_neg        = round(sum(vote_neg_list) / max(len(vote_neg_list), 1), 4)
            vote_neut       = round(sum(vote_neut_list) / max(len(vote_neut_list), 1), 4)
            vote_abs_mean   = round(sum(vote_abs_list) / max(len(vote_abs_list), 1), 4)
            reversal_rate   = round(sum(reversal_list) / max(len(reversal_list), 1), 4)
            net_displacement = round(sum(net_disp_list) / max(len(net_disp_list), 1), 2)
            w_abs_mean      = round(sum(w_abs_list) / max(len(w_abs_list), 1), 4)
            w_std           = round(sum(w_std_list) / max(len(w_std_list), 1), 4)
            near_zero       = round(sum(near_zero_list) / max(len(near_zero_list), 1), 4)
            idx_std         = round(sum(idx_std_list) / max(len(idx_std_list), 1), 2)
            entropy         = round(sum(entropy_list) / max(len(entropy_list), 1), 4)
            disp_95         = round(sum(disp_95_list) / max(len(disp_95_list), 1), 2)
            momentum_mean   = round(sum(momentum_list) / max(len(momentum_list), 1), 4)

            # ── Scale freeze: unfreeze at threshold step ──
            if step == freeze_steps and freeze_steps > 0:
                for p in scale_params:
                    p.requires_grad = True
                print(f"[DIAGNOSTIC] Scale UN-frozen at step {step}. Comparing vote activity...")

            # ── Trinity noise injection ──
            if step % CONFIG['noise_every'] == 0:
                print(f"[TRINITY] Thermal noise injection at step {step}...")
                for module in model.modules():
                    if isinstance(module, PrimeLinear):
                        module.inject_noise(jitter=1)

            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.time() - start_time
            tps = (CONFIG['seq_len'] * accum_steps) / max(elapsed, 1e-6)  # tokens THIS step / time THIS step
            avg_loss = total_loss / step

            stats = {
                'step': step,
                'loss': round(avg_loss, 4),
                'batch_loss': round(loss_val, 4),
                'flips': round(total_flips / 1e6, 4),
                'tps': round(tps, 2),
                'lr': CONFIG['lr'],
                'timestamp': time.time(),
                'scale_frozen': step <= freeze_steps and freeze_steps > 0,
                'migration_rate':  migration_rate,
                'occupancy':       occupancy,
                'vote_pos':        vote_pos,
                'vote_neg':        vote_neg,
                'vote_neut':       vote_neut,
                'vote_abs_mean':   vote_abs_mean,
                'reversal_rate':   reversal_rate,
                'net_displacement': net_displacement,
                'w_abs_mean':      w_abs_mean,
                'w_std':           w_std,
                'near_zero':       near_zero,
                'idx_std':         idx_std,
                'entropy':         entropy,
                'disp_95':         disp_95,
                'momentum_mean':   momentum_mean,
                'baseline_mode':   CONFIG.get('baseline_mode', False)
            }
            history.append(stats)
            all_stats = old_stats + history
            with open(CONFIG['stats_file'], 'w') as f:
                json.dump(all_stats, f)

            print(f"[PRIME] Step {step} {'[SCALE FROZEN]' if step <= freeze_steps and freeze_steps > 0 else ''} "
                  f"{'[BASELINE]' if CONFIG.get('baseline_mode', False) else ''}| "
                  f"Loss: {avg_loss:.4f} | Mig: {migration_rate:.2%} | "
                  f"Disp95: {disp_95:.1f} | Ent: {entropy:.2f} | "
                  f"Mom: {momentum_mean:.2f}")

            # ── Word salad generation ──
            if False: # step % 5 == 0:
                model.eval()
                with torch.no_grad():
                    prompt = "### Instruction:\ndef fibonacci(n):\n### Response:\n"
                    p_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(CONFIG['device'])
                    try:
                        gen = model.generate(p_ids, max_new_tokens=60,
                                             temperature=0.8, do_sample=True,
                                             pad_token_id=tokenizer.eos_token_id)
                        text = tokenizer.decode(gen[0][p_ids.shape[1]:], skip_special_tokens=True)
                        samples.append({'step': step, 'text': text})
                        with open(CONFIG['samples_file'], 'w') as f:
                            json.dump(samples[-20:], f)
                        print(f"[SALAD] {text[:120]}")
                    except Exception as e:
                        print(f"[SALAD] gen failed: {e}")
                model.train()

            # ── Checkpoint ──
            if step % 50 == 0:
                ckpt = f"{CONFIG['checkpoint_prefix']}_{step}.pt"
                torch.save({
                    'step': step,
                    'state_dict': {k: v for k, v in model.state_dict().items()},
                    'stats': stats,
                }, ckpt)
                print(f"[CKPT] Saved {ckpt}")

            # Reset
            batch_idx = 0
            total_loss_per_step = loss_val
            start_time = time.time()

    print("\n[PRIME] Training complete!")
    torch.save(model.state_dict(), 'mamba_prime_final.pt')
    print("[PRIME] Final model saved: mamba_prime_final.pt")


if __name__ == '__main__':
    train()
