"""
AdamW Baseline — Scientific Control for PRIME Experiment
=========================================================
Identical architecture to mamba3_prime_native.py (Mamba3LM, same config).
Replaces PrimeLinear with standard nn.Linear + AdamW optimizer.
Same dataset, same step count, same logging format.

Run AFTER the PRIME run completes (or in parallel on a separate GPU):
  python3 mamba3_adamw_baseline.py

Then generate comparison:
  python3 plot_comparison.py
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys, time, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

CONFIG = {
    'd_model':      1024,
    'n_layers':     28,
    'expand':       2,
    'd_state':      16,
    'seq_len':      256,
    'grad_accum':   4,
    'lr':           5e-4,           # AdamW standard LR
    'total_steps':  500,            # Match or exceed PRIME run for fair comparison
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
    'stats_file':   'stats_adamw_baseline.json',
    'samples_file': 'samples_adamw_baseline.json',
    'log_file':     'training_adamw_baseline.log',
}

_tok_path = os.path.expanduser('~/.hf_token')
login(token=open(_tok_path).read().strip() if os.path.exists(_tok_path)
      else os.environ.get('HF_TOKEN', ''))

class LoggerTee:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, 'a')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self): self.terminal.flush()
    def isatty(self): return False

sys.stdout = LoggerTee(CONFIG['log_file'])
sys.stderr = sys.stdout

# ── Pure-PyTorch Mamba SSM (identical to PRIME script) ───────────────────────
class RealMambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

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
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = F.conv1d(x_in.transpose(1,2), self.conv1d.weight, self.conv1d.bias,
                          padding=self.conv1d.padding[0],
                          groups=self.conv1d.groups)[:, :, :x.shape[1]].transpose(1,2)
        x_conv = F.silu(x_conv)
        y = self._scan(x_conv)
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

# ── Dataset (identical to PRIME script) ──────────────────────────────────────
def make_dataset(tokenizer):
    ds = load_dataset('iamtarun/python_code_instructions_18k_alpaca',
                      split='train', streaming=True)
    oo_path = '/home/phil/.gemini/antigravity/scratch/analysis_project/oo_dataset.jsonl'
    oo_chunks = []
    try:
        with open(oo_path) as f:
            for line in f:
                d = json.loads(line)
                if isinstance(d.get('text'), str) and len(d['text']) > 50:
                    oo_chunks.append(d['text'])
        print(f"[DATA] Loaded {len(oo_chunks)} OO chunks")
    except Exception:
        pass
    import random
    ds_iter = iter(ds)
    def gen():
        while True:
            if oo_chunks and random.random() < 0.25:
                text = random.choice(oo_chunks)
            else:
                try:
                    ex = next(ds_iter)
                    inst = ex.get('instruction', '')
                    out  = ex.get('output', ex.get('response', ''))
                    text = f"### Instruction:\n{inst}\n### Response:\n{out}{tokenizer.eos_token}"
                except StopIteration:
                    return
            tok = tokenizer(text, truncation=True, max_length=CONFIG['seq_len'],
                            padding='max_length', return_tensors='pt')
            ids = tok['input_ids'][0]
            yield ids, ids.clone()
    return gen

# ── Word salad generation ─────────────────────────────────────────────────────
def generate_salad(model, tokenizer, step):
    prompts = [
        "### Instruction:\nWrite a Python function to reverse a string.\n### Response:\n",
        "### Instruction:\nWhat is a neural network?\n### Response:\n",
        "### Instruction:\ndef fibonacci(n):\n### Response:\n",
    ]
    model.eval()
    texts = []
    with torch.no_grad():
        for prompt in prompts:
            try:
                p_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(CONFIG['device'])
                inp = p_ids
                for _ in range(80):
                    logits, _ = model(inp)
                    next_tok = logits[:, -1, :].div(0.8).softmax(-1).multinomial(1)
                    inp = torch.cat([inp, next_tok], dim=1)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break
                texts.append(tokenizer.decode(inp[0][p_ids.shape[1]:], skip_special_tokens=True))
            except Exception as e:
                texts.append(f'[gen error: {e}]')
    model.train()
    return texts

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"[ADAMW] Mamba3 AdamW Baseline — PID {os.getpid()}")
    print(f"[ADAMW] Standard gradient descent control run")
    print(f"[ADAMW] lr={CONFIG['lr']}  betas=(0.9,0.95)  wd=0.1")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    print("[INIT] Building model (identical arch to PRIME run, all standard nn.Linear)...")
    model = Mamba3LM(vocab_size)
    model = model.to(CONFIG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total params: {total_params/1e6:.1f}M")
    print(f"[INIT] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # AdamW — reviewer spec: lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    # Cosine schedule with warmup (standard for LM training)
    warmup_steps = 20
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps)) *
                            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / CONFIG['total_steps'])))
    )

    data_gen = make_dataset(tokenizer)

    step       = 0
    batch_idx  = 0
    total_loss = 0.0
    start_time = time.time()
    history    = []
    accum      = CONFIG['grad_accum']
    all_salads = []

    print("[TRAIN] Starting AdamW baseline training...")
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

        (loss / accum).backward()
        total_loss += loss.item()
        batch_idx  += 1

        print('.', end='', flush=True)
        if batch_idx % accum == 0:
            print(f"[{batch_idx}/{accum}] L:{loss.item():.4f}", end=' ', flush=True)

        if batch_idx >= accum:
            step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tps      = (CONFIG['seq_len'] * accum) / max(time.time() - start_time, 1e-6)
            avg_loss = total_loss / accum
            cur_lr   = scheduler.get_last_lr()[0]
            ppl      = math.exp(min(avg_loss, 20))

            print(f"\n[ADAMW] Step {step} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                  f"LR: {cur_lr:.2e} | TPS: {tps:.1f}")

            stats = {
                'step':      step,
                'loss':      round(avg_loss, 4),
                'ppl':       round(ppl, 2),
                'tps':       round(tps, 2),
                'lr':        round(cur_lr, 8),
                'timestamp': time.time(),
            }
            history.append(stats)
            with open(CONFIG['stats_file'], 'w') as f:
                json.dump(history, f)

            if step % 50 == 0:
                torch.save({'step': step, 'state_dict': model.state_dict(), 'stats': stats},
                           f"adamw_mamba3_{step}.pt")
                print(f"[CKPT] Saved adamw_mamba3_{step}.pt")

                texts = generate_salad(model, tokenizer, step)
                salad_entry = {'step': step, 'text': ' | '.join(texts), 'samples': texts}
                all_salads.append(salad_entry)
                with open(CONFIG['samples_file'], 'w') as f:
                    json.dump(all_salads[-20:], f)
                print(f"[SALAD] Step {step}: {texts[0][:120]}")

            batch_idx  = 0
            total_loss = 0.0
            start_time = time.time()
            torch.cuda.empty_cache()

            if step >= CONFIG['total_steps']:
                break

    print("[ADAMW] Baseline training complete.")
