import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# 1. Publication Style
# ─────────────────────────────────────────────────────────────────────────────

def set_publication_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 24,
        'legend.title_fontsize': 24,
        'lines.linewidth': 4.5,
        'lines.markersize': 10,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '-',
        'grid.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'figure.constrained_layout.use': False,
        'savefig.dpi': 800,
    })

# ─────────────────────────────────────────────────────────────────────────────
# 2. Config
# ─────────────────────────────────────────────────────────────────────────────

P_LIST  = [1,         3,         5,         10,        20       ]
# Colors taken directly from tab10 palette used in the ablation script:
# P=10 (OURS)  → tab10[0] (blue)
# No Patience  → tab10[1] (orange)
# P=3          → tab10[2] (green)
# P=5          → tab10[3] (red)
# P=20         → tab10[4] (purple)
_cmap   = plt.get_cmap('tab10')
COLORS  = [_cmap(1), _cmap(2), _cmap(3), _cmap(0), _cmap(4)]   # ordered by P_LIST
MARKERS = ['o',      'o',      'o',      '*',       'o'     ]
MSIZES  = [180,      180,      180,      420,       180     ]

def label_for(P):
    """Plain text labels — avoids math mode so fontweight='bold' applies everywhere."""
    if P == 1:  return 'No Patience'
    if P == 10: return 'P = 10 (OURS)'
    return f'P = {P}'

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load data & compute stopping points
# ─────────────────────────────────────────────────────────────────────────────

df     = pd.read_csv('run_009.csv')
steps  = df['col_0'].values
values = df['col_2'].values

stopping_points = {}
for P in P_LIST:
    best_val = values[0]
    wait = 0
    for i in range(1, len(values)):
        if values[i] < best_val:
            best_val = values[i]
            wait = 0
        else:
            wait += 1
            if wait >= P:
                stopping_points[P] = (steps[i], values[i])
                break

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────────────────

set_publication_style()

fig, ax = plt.subplots(figsize=(12, 8))

# Main curve — 2.0 matches the weight of lines in the other paper plots
ax.plot(steps, values, color='black', linewidth=2.0, zorder=2)

# Scatter markers
for P, color, marker, msize in zip(P_LIST, COLORS, MARKERS, MSIZES):
    if P not in stopping_points:
        continue
    step, val = stopping_points[P]
    ax.scatter(step, val, color=color, marker=marker, s=msize,
               edgecolors='black', linewidth=1.5, zorder=5)

# Inline annotations
# P=1  (red)    → right of dot, small font so "No Patience" clears the border
# P=3  (blue)   → above
# P=5  (green)  → below the dot (overlaps curve above)
# P=10 (orange) → further up to clear the curve
# P=20 (purple) → below the dot (overlaps curve above)
ANNOT = {
    1:  dict(xytext=( 10,  10), fontsize=16, ha='left',   va='bottom'),
    3:  dict(xytext=(  0,  14), fontsize=16, ha='center', va='bottom'),
    5:  dict(xytext=(  0, -20), fontsize=16, ha='center', va='top'),
    10: dict(xytext=(  0,  24), fontsize=16, ha='center', va='bottom'),
    20: dict(xytext=( 18, -26), fontsize=16, ha='center', va='top'),
}

for P, color in zip(P_LIST, COLORS):
    if P not in stopping_points:
        continue
    step, val = stopping_points[P]
    kw = ANNOT[P]
    ax.annotate(
        label_for(P),
        (step, val),
        textcoords='offset points',
        xytext=kw['xytext'],
        ha=kw['ha'], va=kw['va'],
        color=color,
        fontsize=kw['fontsize'],
        fontweight='bold',
        zorder=6,
    )

# Axes — no bold
ax.set_xlabel('Offline Stabilization Step', fontsize=22, labelpad=12)
ax.set_ylabel('DM Estimator',               fontsize=22, labelpad=12)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)

# Legend with correct star size
legend_handles = []
for P, color, marker in zip(P_LIST, COLORS, MARKERS):
    ms = 15 if marker == '*' else 10
    legend_handles.append(
        Line2D([0], [0], marker=marker, color='w',
               markerfacecolor=color, markeredgecolor='black',
               markeredgewidth=1.2, markersize=ms,
               label=label_for(P))
    )

ax.legend(handles=legend_handles, loc='upper right',
          fontsize=16, frameon=True, framealpha=1.0,
          edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('early_stopping_enhanced.pdf', bbox_inches='tight')
plt.savefig('early_stopping_enhanced.png', bbox_inches='tight')
print("Saved.")