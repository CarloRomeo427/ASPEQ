"""
FLOP Comparison Script
======================
1. Fetches final cumulative_flops per algorithm/env from WandB _flops runs
2. Prints a summary table in TFLOPs (std shown only for faspeq variants)
3. Plots cumulative_flops vs steps with log-scale y-axis (shaded std only for faspeq)

Publication style from ablation script applied:
 - set_publication_style() with large fonts, thick lines, clean spines
 - tab10 palette via get_palette()
 - EMA smoothing via exponential_moving_average()
 - Legend: framed, no fancybox, black edge, placed via fig.legend()
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import argparse
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. Styling & Utilities  (copied/adapted from ablation script)
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


def get_palette(algorithms):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    return {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}


def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


# ─────────────────────────────────────────────────────────────────────────────
# 2. Labels & helpers
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    'faspeq_pct10_pi': '#7B2D8B',   # violet  — OURS
    'speq_o2o':        '#1f77b4',   # blue
    'rlpd':            '#ff7f0e',   # orange
    'calql':           '#2ca02c',   # green
    'sacfd':           '#d62728',   # red
}

ALGO_LABELS = {
    'faspeq_pct10_pi': 'OURS',
    'speq_o2o':        'SPEQ O2O',
    'rlpd':            'RLPD',
    'calql':           'Cal-QL',
    'sacfd':           'SACfD',
}


def auto_label(algo):
    if algo in ALGO_LABELS:
        return ALGO_LABELS[algo]
    direct = {
        'iql': 'IQL', 'sac': 'SAC',
    }
    if algo in direct:
        return direct[algo]
    label = algo
    label = label.replace('_o2o', ' O2O')
    label = label.replace('_pi', ' (π)')
    label = label.replace('_td', ' (TD)')
    label = label.replace('pct10', '10%')
    label = label.replace('pct20', '20%')
    label = label.replace('pct', '%')
    label = label.replace('valpat', 'pat=')
    label = label.replace('_', ' ')
    while '  ' in label:
        label = label.replace('  ', ' ')
    parts = label.split()
    result = []
    for p in parts:
        if p.upper() in ['O2O', 'TD', 'RLPD', 'IQL', 'SAC']:
            result.append(p.upper())
        elif p.startswith('(') or p.endswith(')') or p.endswith('%') or p.startswith('pat='):
            result.append(p)
        else:
            result.append(p.upper())
    return ' '.join(result)


def is_faspeq(algo):
    return 'faspeq' in algo.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def build_flops_run_patterns(algo, env, dataset):
    env_lower = env.lower()
    display_cap = env_lower.capitalize()
    base = f"{display_cap}_{dataset}"

    patterns = []
    if algo in ('faspeq_pct10_pi', 'faspeq_pct10_td', 'faspeq_pct20_pi',
                'faspeq_pct20_td', 'faspeq_o2o'):
        patterns.append(f"{algo}_{base}_flops")
    else:
        patterns.append(f"{algo}_{base}_flops")

    base_orig = f"{env}_{dataset}"
    alt = f"{algo}_{base_orig}_flops"
    if alt not in patterns:
        patterns.append(alt)

    return patterns


def fetch_flops_curves(api, entity, project, envs, algorithms, valid_seeds, dataset):
    print("  Querying WandB for _flops runs...")
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": "_flops$"}},
        per_page=200,
    )

    run_index = {}
    for run in runs:
        run_index.setdefault(run.name, []).append(run)
    print(f"  Indexed {sum(len(v) for v in run_index.values())} _flops runs.")

    data = {algo: {} for algo in algorithms}

    for env in envs:
        for algo in algorithms:
            patterns = build_flops_run_patterns(algo, env, dataset)
            seed_curves = []

            for pattern in patterns:
                for run in run_index.get(pattern, []):
                    if valid_seeds is not None:
                        seed = run.config.get('seed', None)
                        if seed not in valid_seeds:
                            continue

                    history = run.history(
                        keys=["cumulative_flops", "_step"], pandas=True
                    )
                    if "cumulative_flops" in history.columns:
                        h = history.dropna(subset=["cumulative_flops"])
                        if len(h) > 0:
                            seed_curves.append((
                                h["_step"].values,
                                h["cumulative_flops"].values,
                            ))
                            print(f"  {pattern} seed={run.config.get('seed')}: "
                                  f"final={h['cumulative_flops'].values[-1]:.3e}")

            data[algo][env] = seed_curves if seed_curves else None

    return data


def aggregate_across_seeds(seed_curves):
    if not seed_curves:
        return None
    all_steps = set()
    for steps, _ in seed_curves:
        all_steps.update(steps)
    all_steps = sorted(all_steps)

    interpolated = [np.interp(all_steps, s, f) for s, f in seed_curves]
    stacked = np.array(interpolated)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
    return np.array(all_steps), mean, std


# ─────────────────────────────────────────────────────────────────────────────
# 4. Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_flops_table(flops_data, algorithms, envs):
    label_map = {a: auto_label(a) for a in algorithms}

    print("\n" + "=" * 100)
    print(f"{'Algorithm':25s} | {'Env':30s} | {'Final TFLOPs':>22s} | {'Seeds':>5s}")
    print("-" * 100)

    algo_totals = {a: [] for a in algorithms}
    algo_stds   = {a: [] for a in algorithms}

    for algo in algorithms:
        for env in envs:
            curves = flops_data[algo].get(env)
            if curves is None:
                continue
            finals = [flops[-1] for _, flops in curves]
            mean_tflops = np.mean(finals) / 1e12
            std_tflops  = (np.std(finals, ddof=1) if len(finals) > 1 else 0.0) / 1e12
            algo_totals[algo].append(mean_tflops)
            algo_stds[algo].append(std_tflops)

            val_str = (f"{mean_tflops:.1f} ± {std_tflops:.1f}"
                       if is_faspeq(algo) else f"{mean_tflops:.1f}")
            print(f"{label_map[algo]:25s} | {env:30s} | {val_str:>22s} | {len(finals):5d}")

    print("=" * 100)
    print(f"\n{'Algorithm':25s} | {'Mean TFLOPs (across envs)':>34s}")
    print("-" * 65)
    for algo in algorithms:
        vals = algo_totals[algo]
        stds = algo_stds[algo]
        if vals:
            if is_faspeq(algo):
                mean_of_means = np.mean(vals)
                combined_std  = np.sqrt(np.mean(np.array(stds) ** 2))
                val_str = f"{mean_of_means:.1f} ± {combined_std:.1f}"
            else:
                val_str = f"{np.mean(vals):.1f}"
            print(f"{label_map[algo]:25s} | {val_str:>34s}")
        else:
            print(f"{label_map[algo]:25s} | {'N/A':>34s}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregated log-scale FLOPs plot  (publication style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_flops_log(flops_data, algorithms, envs, dataset, ema_alpha=0.05, std_alpha=0.15):
    label_map = {a: auto_label(a) for a in algorithms}

    fig, ax = plt.subplots(figsize=(12, 8))

    for algo in algorithms:
        all_curves = []
        for env in envs:
            curves = flops_data[algo].get(env)
            if curves is None:
                continue
            agg = aggregate_across_seeds(curves)
            if agg is not None:
                all_curves.append(agg)  # (steps, mean, std)

        if not all_curves:
            continue

        max_step = max(s.max() for s, _, _ in all_curves)
        common_steps = np.linspace(0, max_step, 500)
        interp_means = [np.interp(common_steps, s, m) for s, m, _ in all_curves]
        interp_stds  = [np.interp(common_steps, s, sd) for s, _, sd in all_curves]

        mean_flops   = np.mean(interp_means, axis=0)
        combined_std = np.sqrt(np.mean(np.array(interp_stds) ** 2, axis=0))

        mean_tflops = mean_flops / 1e12
        std_tflops  = combined_std / 1e12

        # EMA smoothing
        mean_ema = exponential_moving_average(mean_tflops, alpha=ema_alpha)
        std_ema  = exponential_moving_average(std_tflops,  alpha=ema_alpha)

        color = ALGO_COLORS.get(algo, '#888888')
        ax.plot(common_steps, mean_ema, label=label_map[algo],
                linewidth=4.5, color=color)

        if is_faspeq(algo):
            ax.fill_between(
                common_steps,
                np.maximum(mean_ema - std_ema, 1e-3),
                mean_ema + std_ema,
                alpha=std_alpha,
                color=color,
                linewidth=0,
            )

    ax.set_yscale('log')
    ax.set_ylim(bottom=10)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel("Environment Steps", fontsize=22, labelpad=12)
    ax.set_ylabel("Cumulative TFLOPs (log scale)", fontsize=22, labelpad=12)
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4, which='both')

    # Legend — bottom right, smaller, framed
    legend_handles = [Line2D([0], [0], color=ALGO_COLORS.get(a, '#888888'), lw=4.5)
                      for a in algorithms]
    legend_labels  = [label_map[a] for a in algorithms]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower right',
        fontsize=16,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
        columnspacing=1.2,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-environment subplot figure  (publication style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_flops_per_env(flops_data, algorithms, envs, dataset, ema_alpha=0.05, std_alpha=0.15):
    label_map = {a: auto_label(a) for a in algorithms}

    n      = len(envs)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 6 * n_rows),
                             sharey=False)
    axes = np.atleast_2d(axes).flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        for algo in algorithms:
            curves = flops_data[algo].get(env)
            if curves is None:
                continue
            agg = aggregate_across_seeds(curves)
            if agg is None:
                continue
            steps, mean, std = agg

            tflops     = mean / 1e12
            std_tflops = std  / 1e12

            tflops_ema     = exponential_moving_average(tflops,     alpha=ema_alpha)
            std_tflops_ema = exponential_moving_average(std_tflops, alpha=ema_alpha)

            color = ALGO_COLORS.get(algo, '#888888')
            ax.plot(steps, tflops_ema, label=label_map[algo],
                    linewidth=4.5, color=color)

            if is_faspeq(algo):
                ax.fill_between(
                    steps,
                    np.maximum(tflops_ema - std_tflops_ema, 1e-3),
                    tflops_ema + std_tflops_ema,
                    alpha=std_alpha,
                    color=color,
                    linewidth=0,
                )

        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4, which='both')

        # Only left-most column gets y-label
        if idx % n_cols == 0:
            ax.set_ylabel("TFLOPs (log scale)", fontsize=22, labelpad=12)
        ax.set_xlabel("Steps", fontsize=22, labelpad=12)
        ax.text(0.5, 1.02, env.replace("-v5", ""),
                transform=ax.transAxes,
                ha='center', va='bottom', fontsize=24, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    # Single shared legend — bottom right of figure, smaller, framed
    legend_handles = [Line2D([0], [0], color=ALGO_COLORS.get(a, '#888888'), lw=4.5)
                      for a in algorithms]
    legend_labels  = [label_map[a] for a in algorithms]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower right',
        fontsize=16,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
        columnspacing=1.2,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert",
                        choices=["expert", "medium", "simple"])
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--std-alpha", type=float, default=0.15)
    parser.add_argument("--per-env", action="store_true",
                        help="Also generate per-environment subplot figure")
    args = parser.parse_args()

    set_publication_style()

    entity      = "carlo-romeo-alt427"
    project     = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}

    algorithms = [
        "rlpd",
        "calql",
        "speq_o2o",
        "faspeq_pct10_pi",
        "sacfd",
    ]

    all_envs = [
        "Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
        "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5",
        "Reacher-v5", "Swimmer-v5",
    ]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    envs = simple_only_envs if args.dataset == "simple" else all_envs

    os.makedirs("Plots", exist_ok=True)
    api = wandb.Api()

    print(f"Fetching _flops runs for {args.dataset} dataset...")
    flops_data = fetch_flops_curves(
        api, entity, project, envs, algorithms, valid_seeds, args.dataset
    )

    # 1. Summary table
    print_flops_table(flops_data, algorithms, envs)

    # 2. Aggregated log-scale plot
    fig = plot_flops_log(flops_data, algorithms, envs, args.dataset,
                         args.ema_alpha, args.std_alpha)
    plt.savefig(f"Plots/flops_logscale_{args.dataset}.pdf", bbox_inches="tight")
    plt.savefig(f"Plots/flops_logscale_{args.dataset}.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: Plots/flops_logscale_{args.dataset}.pdf / .png")

    # 3. Per-env plots (optional)
    if args.per_env:
        fig = plot_flops_per_env(flops_data, algorithms, envs, args.dataset,
                                 args.ema_alpha, args.std_alpha)
        plt.savefig(f"Plots/flops_per_env_{args.dataset}.pdf", bbox_inches="tight")
        plt.savefig(f"Plots/flops_per_env_{args.dataset}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: Plots/flops_per_env_{args.dataset}.pdf / .png")

    print("\nDONE")