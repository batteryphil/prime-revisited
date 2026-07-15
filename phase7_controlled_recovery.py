"""
Phase 7: Controlled Recovery
Fixes: Thawing the backbone while explicitly FREEZING the MIMO experts to prevent catastrophic mode collapse.
"""
import os, sys, time, json, math, glob
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import AutoTokenizer
from mamba_ssm import Mamba

# ── Inherit unchanged pieces from phase5 ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from mamba3_titan_mimo_phase5 import (
    build_prime_lut, PrimeLinear, ContinuousVoteOptimizer,
    MambaLayer, TitanMIMO, CONFIG, make_dataset,
    run_validation, check_thermal_throttle, LoggerTee
)

# Phase 6 overrides
CONFIG['log_file']     = 'training_titan_mimo_phase7.log'
CONFIG['total_steps']  = 10_000_000  # unlimited — run until manually stopped

# ── Patch: amplified gradient hook for out_proj ───────────────────────────────
OUTPROJ_GRAD_SCALE = 20.0  # scale gradients 20x to overcome noise floor

def _amplified_vote_hook(self, grad):
    """Same as normal hook but scales gradient 20x before signing."""
    with torch.no_grad():
        sign_t = (grad * OUTPROJ_GRAD_SCALE).sign().to(torch.int8)
        if self.step_counter[0] % 50 == 0:
            agreement = (sign_t == self.last_grad_sign).float().mean()
            self.sign_agreement_ema[0] = self.sign_agreement_ema[0] * 0.95 + agreement * 0.05
        self.last_grad_sign.copy_(sign_t)
        pressure = sign_t.mul_(10).to(torch.int16)
        self.vote_buffer.add_(pressure).clamp_(-32760, 32760)

def patch_outproj_layer(prime_linear):
    """Reset PI state and attach amplified hook."""
    import types
    # 1. Reset PI controller completely
    prime_linear.log_threshold[0] = math.log(12.0)   # back to default
    prime_linear.pi_integral[0]   = 0.0
    prime_linear.ema_flip_rate[0] = 0.0
    prime_linear.vote_buffer.zero_()
    prime_linear.step_counter[0]  = 0                  # restart warmup counter

    # 2. Much lower target: 1% (was 5%) — PI won't panic and slam threshold up
    prime_linear.target_flip_rate = 0.01

    # 3. Widen PI controller ceiling to allow easier flipping
    # Clamp log_threshold max to log(32) instead of log(128)
    # We patch apply_votes logic by monkey-patching the clamp
    _orig_apply = prime_linear.apply_votes.__func__

    def _patched_apply(self_pl, lr=1e-4, step=None):
        # Same as original but clamp ceiling at log(32) not log(128)
        bs      = self_pl.BLOCK_SIZE
        flat    = self_pl.vote_buffer.view(-1)
        n       = flat.numel()
        aligned = n - (n % bs)
        flat_a  = flat[:aligned]

        blocks    = flat_a.float().view(-1, bs)
        magnitude = blocks.abs().mean(dim=1, keepdim=True)

        self_pl.step_counter[0] += 1
        threshold  = torch.exp(self_pl.log_threshold[0])
        authorized = (magnitude * (lr / 1e-4) >= threshold)

        combined = (self_pl.base_idx.view(-1)[:aligned].to(torch.int32) * 256 +
                    self_pl.fine_idx.view(-1)[:aligned].to(torch.int32))

        vote_pos = (flat_a > 0).float().mean()
        vote_neg = (flat_a < 0).float().mean()
        with torch.no_grad():
            hist = torch.bincount(self_pl.vote_buffer.abs().view(-1), minlength=32761)
            entropy = {'zeros': hist[0].item(), 'low': hist[1:4].sum().item(),
                       'med': hist[4:12].sum().item(), 'high': hist[12:].sum().item()}
        vpu = (self_pl.vote_buffer.float().abs().mean() / threshold).item()

        if step is not None and step % 50 == 0:
            self_pl.rolling_flips[self_pl.rolling_idx[0]] = self_pl.current_interval_flips[0]
            self_pl.rolling_idx[0] = (self_pl.rolling_idx[0] + 1) % 200
            self_pl.current_interval_flips[0] = 0

        retention_pressure = self_pl.rolling_flips.sum().item() + self_pl.current_interval_flips[0].item()

        if not authorized.any():
            return {'flips': 0, 'vote_pos': vote_pos.item(), 'vote_neg': vote_neg.item(),
                    'name': self_pl.name, 'entropy': entropy, 'vpu': vpu, 'reversals': 0,
                    'numel': n, 'retention_pressure': retention_pressure}

        step_dir  = torch.sign(blocks)
        block_dir = step_dir.mean(dim=1, keepdim=True).sign().to(torch.int8)
        auth_sq   = authorized.squeeze()
        ld_sq     = self_pl.last_dir.squeeze()
        bd_sq     = block_dir.squeeze()

        reversed_blocks = auth_sq & (bd_sq != ld_sq) & (ld_sq != 0)
        same_dir        = (bd_sq == ld_sq).to(torch.int8)
        new_momentum    = torch.clamp((self_pl.momentum.squeeze() * same_dir + same_dir).to(torch.int8), 0, 8)
        self_pl.momentum = new_momentum.view(-1, 1)

        cblocks     = combined.view(-1, bs)
        center_dist = (cblocks - 32768).float().abs().mean(dim=1, keepdim=True)
        base_stride = torch.clamp((self_pl.MAX_STRIDE * (1.0 - center_dist / 32768.0)).long(), min=1)
        dyn_stride  = torch.clamp(base_stride * (1 + self_pl.momentum.float() / 2.0).to(torch.long), max=self_pl.MAX_STRIDE)
        dyn_stride[reversed_blocks.view(-1, 1)] = 0

        update      = (authorized.float() * step_dir * dyn_stride).to(torch.int32)
        moved       = authorized & ~reversed_blocks.unsqueeze(-1).expand_as(authorized)
        total_flips = moved.sum() * bs

        new_combined = torch.clamp(combined - update.view(-1), 0, 65535)
        self_pl.base_idx.copy_(torch.div(new_combined, 256, rounding_mode='floor').to(torch.uint8).view(self_pl.base_idx.shape))
        self_pl.fine_idx.copy_((new_combined % 256).to(torch.uint8).view(self_pl.fine_idx.shape))
        self_pl.last_dir = block_dir.clone()

        self_pl.vote_buffer.view(-1)[:aligned].view(-1, bs).masked_fill_(authorized, 0)
        with torch.no_grad():
            hist = torch.bincount(self_pl.vote_buffer.abs().view(-1), minlength=32761)
            entropy = {'zeros': hist[0].item(), 'low': hist[1:4].sum().item(),
                       'med': hist[4:12].sum().item(), 'high': hist[12:].sum().item()}
        self_pl.vote_buffer = (self_pl.vote_buffer.float() * self_pl.DECAY_RATE).to(torch.int16)

        flips     = total_flips.item()
        reversals = reversed_blocks.sum().item() * bs

        # PI controller with LOWERED ceiling (log(32) instead of log(128))
        with torch.no_grad():
            current_flip_rate = flips / max(n, 1)
            self_pl.ema_flip_rate[0] = self_pl.ema_flip_rate[0] * 0.999 + current_flip_rate * 0.001
            if self_pl.step_counter[0] > 200 and self_pl.step_counter[0] % 50 == 0:
                target    = self_pl.target_flip_rate
                error     = (self_pl.ema_flip_rate[0] - target) / max(target, 1e-3)
                kp, ki    = 0.3, 0.005   # gentler PI gains
                self_pl.pi_integral[0] += error
                adjustment = (kp * error) + (ki * self_pl.pi_integral[0])
                new_log    = self_pl.log_threshold[0] + adjustment
                # Ceiling at log(32) — prevents the runaway freeze
                clamped    = torch.clamp(new_log, math.log(2.0), math.log(32.0))
                self_pl.delta_log_thresh[0] = clamped - self_pl.log_threshold[0]
                self_pl.log_threshold[0]    = clamped
                if clamped.item() >= math.log(32.0) or clamped.item() <= math.log(2.0):
                    self_pl.pi_integral[0] = 0.0

        if step is not None:
            self_pl.current_interval_flips[0] += flips

        return {'flips': flips, 'vote_pos': vote_pos.item(), 'vote_neg': vote_neg.item(),
                'name': self_pl.name, 'entropy': entropy, 'vpu': vpu,
                'reversals': reversals, 'numel': n,
                'retention_pressure': retention_pressure,
                'ema_flip': self_pl.ema_flip_rate[0].item(),
                'sign_agreement': self_pl.sign_agreement_ema[0].item(),
                'threshold': torch.exp(self_pl.log_threshold[0]).item(),
                'delta_thresh': self_pl.delta_log_thresh[0].item()}

    prime_linear.apply_votes = types.MethodType(_patched_apply, prime_linear)
    print(f"  [P7] Patched {prime_linear.name}: threshold reset to 12, target→1%, ceiling→32, grad_scale=20x")


def patched_forward(self, input_ids, labels=None, domain_id=None):
    x = self.embedding(input_ids)
    for layer in self.layers[:self._mid]:
        x = layer(x)

    B, L, _ = x.shape
    router_x = F.layer_norm(x.detach(), (x.size(-1),))
    router_logits = self.domain_router(router_x)
    
    temperature = 1.0 if self.training else 0.8
    route_weights = F.softmax(router_logits / temperature, dim=-1)
    route_weights_detached = route_weights.detach()

    raw_arm_outs = []
    for arm in self.mimo_arms:
        raw_arm_outs.append(arm(x))
    
    stacked = torch.stack(raw_arm_outs, dim=-1)
    collapsed = torch.einsum('b l d m, b l m -> b l d', stacked, route_weights_detached.to(stacked.dtype))
    x = x + collapsed

    for layer in self.layers[self._mid:]:
        x = layer(x)

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
            target[input_ids == 0] = -100
            routing_loss = F.cross_entropy(
                router_logits.view(-1, 4),
                target.reshape(-1),
                ignore_index=-100
            )
            loss = loss + (0.5 * routing_loss)
    return logits, loss

TitanMIMO.forward = patched_forward

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', default=CONFIG['corpus_path'])
    parser.add_argument('--val_path',    default=CONFIG.get('val_path', './data/titan_val_suite.jsonl'))
    parser.add_argument('--ckpt_dir',    default=CONFIG['ckpt_dir'])
    args = parser.parse_args()
    CONFIG['corpus_path'] = args.corpus_path
    CONFIG['val_path']    = args.val_path
    CONFIG['ckpt_dir']    = args.ckpt_dir

    sys.stdout = LoggerTee(CONFIG['log_file'])
    sys.stderr = sys.stdout

    print("\n" + "="*60)
    print("[PHASE 7] Controlled Recovery — Titan MIMO 650M")
    print("="*60)

    lut = build_prime_lut()
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TitanMIMO()

    # Wrap with PrimeLinear (same as phase5)
    for idx, layer in enumerate(model.layers):
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj,  lut, name=f"layers.{idx}.ssm.in_proj",  target_flip_rate=0.05)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut, name=f"layers.{idx}.ssm.out_proj", target_flip_rate=0.05)
    arm_names = ['chat', 'math', 'code', 'tool']
    for idx, arm in enumerate(model.mimo_arms):
        aname = arm_names[idx]
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj,  lut, name=f"mimo_arms.{idx}.ssm.in_proj",  target_flip_rate=0.05)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut, name=f"mimo_arms.{idx}.ssm.out_proj", target_flip_rate=0.05)
    model.domain_router = PrimeLinear(model.domain_router, lut, name="domain_router", target_flip_rate=0.05)

    # Load latest checkpoint
    ckpt_dir  = CONFIG['ckpt_dir']
    latest    = os.path.join(ckpt_dir, "titan_mimo_phase5.pt")
    print(f"[INIT] Loading checkpoint: {latest}")
    ckpt_data = torch.load(latest, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt_data['state_dict'], strict=False)
    step = ckpt_data.get('step', 0)
    print(f"[INIT] Resumed from step {step}")

    model = model.to(CONFIG['device'])

    # ── Phase 7 Targeted Freezing ─────────────────────────────────────────────
    print("\n[P7] Freezing MIMO Router, Expert Arms, and LM Head...")
    for p in model.domain_router.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False
    for arm in model.mimo_arms:
        for p in arm.parameters():
            p.requires_grad = False

    # ── Apply Phase 7 patches to ALL out_proj layers ──────────────────────────
    print("\n[P7] Patching out_proj layers...")
    outproj_count = 0
    for name, module in model.named_modules():
        if isinstance(module, PrimeLinear) and 'out_proj' in module.name:
            patch_outproj_layer(module)
            # Attach amplified gradient hook (replaces weight property hook)
            import types
            module._vote_hook = types.MethodType(_amplified_vote_hook, module)
            outproj_count += 1
    print(f"[P7] Patched {outproj_count} out_proj layers. Starting training...\n")

    # Continuous optimizer
    continuous_params = [p for n, p in model.named_parameters()
                         if 'base_idx' not in n and 'fine_idx' not in n and 'vote_buffer' not in n and p.requires_grad]
    optimizer = ContinuousVoteOptimizer(continuous_params, lr=1e-4)
    if 'optimizer' in ckpt_data:
        try:
            optimizer.load_state_dict(ckpt_data['optimizer'])
            print("[INIT] Continuous optimizer state loaded.")
        except Exception as e:
            print(f"[INIT] Optimizer state mismatch — reinitializing. ({e})")

    data_gen    = make_dataset(tokenizer)
    batch_idx   = 0
    total_loss  = 0.0
    start_time  = time.time()
    accum       = CONFIG['grad_accum']
    interval_flips = 0
    model.train()

    print("[TRAIN] Phase 7 training loop started...")
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

        if batch_idx >= accum:
            step += 1

            total_flips = 0
            vpos_sum = 0.0
            vneg_sum = 0.0
            count    = 0
            
            total_entropy = {'zeros': 0, 'low': 0, 'med': 0, 'high': 0}
            total_reversals = 0
            total_unique = 0
            layer_stats = []

            for m in model.modules():
                if isinstance(m, PrimeLinear):
                    # [Phase 7.1] The True Freeze: explicitly block MIMO/router from voting
                    if 'mimo_arms' in m.name or 'router' in m.name:
                        continue
                    
                    r = m.apply_votes(lr=CONFIG['lr'], step=step)
                    flips = r['flips'] if not hasattr(r['flips'], 'item') else r['flips'].item()
                    total_flips += flips
                    vpos_sum    += r['vote_pos']
                    vneg_sum    += r['vote_neg']
                    count       += 1
                    
                    total_reversals += r.get('reversals', 0)
                    total_unique += flips / m.BLOCK_SIZE
                    if 'entropy' in r:
                        total_entropy['zeros'] += r['entropy']['zeros']
                        total_entropy['low'] += r['entropy']['low']
                        total_entropy['med'] += r['entropy']['med']
                        total_entropy['high'] += r['entropy']['high']
                    
                    layer_stats.append({
                        'name': r['name'],
                        'rate': r.get('ema_flip', 0.0) * 100,
                        'thresh': r.get('threshold', 0.0)
                    })

            c_flips = optimizer.step()
            c_flips_int = int(c_flips.item()) if hasattr(c_flips, 'item') else int(c_flips)
            total_flips += c_flips_int
            interval_flips += total_flips

            for p in model.parameters():
                p.grad = None

            tps      = (CONFIG['seq_len'] * CONFIG['batch_size'] * accum) / max(time.time() - start_time, 1e-6)
            avg_loss = total_loss / accum
            mv       = vpos_sum / max(count, 1)
            
            total_params = sum(total_entropy.values()) + 1e-8
            sorted_layers = sorted(layer_stats, key=lambda x: x['rate'], reverse=True)
            
            telemetry_data = {
                "step": step,
                "loss": avg_loss,
                "entropy": {
                    "zeros": total_entropy['zeros']/total_params*100,
                    "low": total_entropy['low']/total_params*100,
                    "med": total_entropy['med']/total_params*100,
                    "high": total_entropy['high']/total_params*100
                },
                "global_flips": total_flips,
                "continuous_flips": c_flips_int,
                "total_continuous_params": sum(p.numel() for p in continuous_params),
                "unique_flipped": int(total_unique),
                "total_reversals": total_reversals,
                "top_layers": sorted_layers[:5],
                "bottom_layers": sorted_layers[-5:],
                "all_layers": sorted_layers
            }
            with open("telemetry_live.json", "w") as f:
                import json
                json.dump(telemetry_data, f)

            print(f"[P7] Step {step} | Loss: {avg_loss:.4f} | Flips: {total_flips} | TPS: {tps:.1f}")

            # Detailed diagnostics every 100 steps
            if step % 100 == 0:
                print("\n[P7-DIAG] out_proj flip health:")
                for name, module in model.named_modules():
                    if isinstance(module, PrimeLinear) and 'out_proj' in module.name:
                        thresh = math.exp(module.log_threshold[0].item())
                        rate   = module.ema_flip_rate[0].item() * 100
                        agmt   = module.sign_agreement_ema[0].item() * 100
                        print(f"  {name:40s} | thresh:{thresh:6.1f} | flip:{rate:5.2f}% | sign_agmt:{agmt:5.1f}%")

                # Save live checkpoint
                live_path = os.path.join(ckpt_dir, "titan_mimo_phase7.pt")
                torch.save({'step': step, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, live_path)
                print(f"[SAVE] Step {step} → {live_path}")

            if step % 5000 == 0:
                run_validation(model, tokenizer, step, interval_flips)
                interval_flips = 0
                arch_path = os.path.join(ckpt_dir, f"titan_mimo_p7_{step}.pt")
                torch.save({'step': step, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, arch_path)

            batch_idx  = 0
            total_loss = 0.0
            start_time = time.time()
            torch.cuda.empty_cache()
