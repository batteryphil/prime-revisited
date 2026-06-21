"""
PRIME vs AdamW Comparison Plot
================================
Loads stats_mamba3_native.json and stats_adamw_baseline.json
and generates a multi-panel comparison figure.

Usage:
    python3 plot_comparison.py
    python3 plot_comparison.py --steps 182   # compare only first N steps
"""
import json, math, sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=None, help='Limit to first N steps')
parser.add_argument('--out',   type=str, default='loss_comparison.png')
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path, max_steps=None):
    try:
        with open(path) as f:
            data = json.load(f)
        if max_steps:
            data = [d for d in data if d['step'] <= max_steps]
        return data
    except FileNotFoundError:
        print(f"[WARN] {path} not found — skipping")
        return []
    except Exception as e:
        print(f"[WARN] {path} load error: {e}")
        return []

prime = load('stats_mamba3_native.json',  args.steps)
adamw = load('stats_adamw_baseline.json', args.steps)

if not prime and not adamw:
    print("No data files found. Run training first.")
    sys.exit(1)

# ── Rolling average helper ────────────────────────────────────────────────────
def rolling(vals, w=10):
    out = []
    for i in range(len(vals)):
        window = vals[max(0, i-w+1):i+1]
        out.append(sum(window) / len(window))
    return out

# ── Color scheme ─────────────────────────────────────────────────────────────
PRIME_COLOR = '#7c6af7'   # purple
ADAMW_COLOR = '#3de0a0'   # teal
BG          = '#0a0b0f'
PANEL_BG    = '#111318'
GRID_COLOR  = '#1e2130'
TEXT_COLOR  = '#e2e6f0'
MUTED       = '#5a6080'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_COLOR,
    'axes.labelcolor':   TEXT_COLOR,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'grid.color':        GRID_COLOR,
    'grid.linewidth':    0.5,
    'text.color':        TEXT_COLOR,
    'font.family':       'monospace',
    'font.size':         10,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle('PRIME vs AdamW — Mamba3 Cold-Start Training',
             color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_loss   = fig.add_subplot(gs[0, :2])   # Main loss curve — wide
ax_ppl    = fig.add_subplot(gs[0, 2])    # Perplexity
ax_mig    = fig.add_subplot(gs[1, 0])    # Migration rate (PRIME only)
ax_disp   = fig.add_subplot(gs[1, 1])    # Disp95 (PRIME only)
ax_ent    = fig.add_subplot(gs[1, 2])    # Entropy (PRIME only)

def style_ax(ax, title, xlabel='Step', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Loss curve ────────────────────────────────────────────────────────────────
style_ax(ax_loss, 'Cross-Entropy Loss — Raw & Rolling Avg (window=10)', ylabel='Loss')

if prime:
    steps_p = [d['step'] for d in prime]
    loss_p  = [d['loss'] for d in prime]
    roll_p  = rolling(loss_p)
    ax_loss.plot(steps_p, loss_p, color=PRIME_COLOR, alpha=0.25, linewidth=0.8, label='_')
    ax_loss.plot(steps_p, roll_p, color=PRIME_COLOR, linewidth=2.0, label=f'PRIME (final={loss_p[-1]:.3f})')

if adamw:
    steps_a = [d['step'] for d in adamw]
    loss_a  = [d['loss'] for d in adamw]
    roll_a  = rolling(loss_a)
    ax_loss.plot(steps_a, loss_a, color=ADAMW_COLOR, alpha=0.25, linewidth=0.8, label='_')
    ax_loss.plot(steps_a, roll_a, color=ADAMW_COLOR, linewidth=2.0, label=f'AdamW (final={loss_a[-1]:.3f})')

if not adamw:
    ax_loss.text(0.5, 0.5, 'AdamW baseline not yet run\nRun: python3 mamba3_adamw_baseline.py',
                 transform=ax_loss.transAxes, ha='center', va='center',
                 color=MUTED, fontsize=9, style='italic')

ax_loss.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
ax_loss.set_ylim(bottom=0)

# ── Perplexity ────────────────────────────────────────────────────────────────
style_ax(ax_ppl, 'Perplexity (exp(loss), rolling)', ylabel='PPL')

if prime:
    ppl_p = [math.exp(min(l, 15)) for l in rolling(loss_p)]
    ax_ppl.plot(steps_p, ppl_p, color=PRIME_COLOR, linewidth=1.8, label='PRIME')
    ax_ppl.set_yscale('log')

if adamw:
    ppl_a = [math.exp(min(l, 15)) for l in rolling(loss_a)]
    ax_ppl.plot(steps_a, ppl_a, color=ADAMW_COLOR, linewidth=1.8, label='AdamW')
    ax_ppl.set_yscale('log')

ax_ppl.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

# ── PRIME-specific: Migration rate ────────────────────────────────────────────
style_ax(ax_mig, 'Migration Rate (PRIME)', ylabel='Mig %')
if prime:
    mig_p = [d.get('migration_rate', 0) for d in prime]
    ax_mig.plot(steps_p, mig_p, color=PRIME_COLOR, linewidth=1.5)
    ax_mig.axhline(y=19.3, color='#f7a94a', linewidth=1, linestyle='--', alpha=0.7,
                   label='Phase 1 proven baseline (19.3%)')
    ax_mig.fill_between(steps_p, mig_p, alpha=0.15, color=PRIME_COLOR)
    ax_mig.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)
else:
    ax_mig.text(0.5, 0.5, 'No PRIME data', transform=ax_mig.transAxes,
                ha='center', va='center', color=MUTED, fontsize=9)

# ── PRIME-specific: Displacement ─────────────────────────────────────────────
style_ax(ax_disp, 'Disp95 — LUT Walk Distance (PRIME)', ylabel='Index steps from init')
if prime:
    disp_p = [d.get('disp_95', 0) for d in prime]
    ax_disp.plot(steps_p, disp_p, color='#f7a94a', linewidth=1.5)
    ax_disp.fill_between(steps_p, disp_p, alpha=0.15, color='#f7a94a')
    # Fit and show trend line
    z = np.polyfit(steps_p, disp_p, 1)
    trend = np.poly1d(z)
    ax_disp.plot(steps_p, trend(steps_p), '--', color=MUTED, linewidth=1,
                 label=f'trend: {z[0]:.2f}/step')
    ax_disp.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)

# ── PRIME-specific: Entropy ───────────────────────────────────────────────────
style_ax(ax_ent, 'LUT Entropy (index spread, PRIME)', ylabel='bits')
if prime:
    ent_p = [d.get('entropy', 0) for d in prime]
    ax_ent.plot(steps_p, ent_p, color='#f7524a', linewidth=1.5)
    ax_ent.axhline(y=16.0, color=MUTED, linewidth=0.8, linestyle=':', alpha=0.5,
                   label='max (16 bits)')
    ax_ent.fill_between(steps_p, ent_p, alpha=0.15, color='#f7524a')
    ax_ent.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)

# ── Annotations ──────────────────────────────────────────────────────────────
if prime and adamw:
    min_p = min(loss_p)
    min_a = min(loss_a)
    fig.text(0.5, 0.01,
             f'PRIME best: {min_p:.4f} @ step {prime[loss_p.index(min_p)]["step"]}  |  '
             f'AdamW best: {min_a:.4f} @ step {adamw[loss_a.index(min_a)]["step"]}  |  '
             f'Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='center', color=MUTED, fontsize=8)
elif prime:
    min_p = min(loss_p)
    fig.text(0.5, 0.01,
             f'PRIME: {len(prime)} steps  |  best loss: {min_p:.4f}  |  '
             f'AdamW baseline not yet available  |  '
             f'Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
             ha='center', color=MUTED, fontsize=8)

plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"[PLOT] Saved → {args.out}")
print(f"       PRIME steps: {len(prime)}")
print(f"       AdamW steps: {len(adamw)}")
if prime:
    print(f"       PRIME loss range: {min(loss_p):.4f} – {max(loss_p):.4f}")
if adamw:
    print(f"       AdamW loss range: {min(loss_a):.4f} – {max(loss_a):.4f}")
