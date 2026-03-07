"""
FLOP Comparison Script — HalfCheetah-v5 Expert
================================================
- Baselines (rlpd, calql, speq_o2o, sacfd): seed 696969 only
- faspeq_pct10_pi: seeds {696969, 42, 0} with mean ± std

Publication style from ablation script.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import argparse
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. Styling & Utilities
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
    return algo.upper()


def is_faspeq(algo):
    return 'faspeq' in algo.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def build_flops_run_patterns(algo, env, dataset):
    env_lower = env.lower()
    display_cap = env_lower.capitalize()
    base = f"{display_cap}_{dataset}"

    patterns = [f"{algo}_{base}_flops"]

    base_orig = f"{env}_{dataset}"
    alt = f"{algo}_{base_orig}_flops"
    if alt not in patterns:
        patterns.append(alt)

    return patterns


def fetch_flops_curves(api, entity, project, envs, algorithms, seeds_per_algo, dataset):
    """
    Fetch _flops runs from WandB.

    Args:
        seeds_per_algo: dict mapping algo -> set of valid seeds.
    """
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
            algo_seeds = seeds_per_algo.get(algo, set())
            patterns = build_flops_run_patterns(algo, env, dataset)
            seed_curves = []

            for pattern in patterns:
                for run in run_index.get(pattern, []):
                    seed = run.config.get('seed', None)
                    if seed not in algo_seeds:
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
                            print(f"    {pattern} seed={seed}: "
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
# 5. Aggregated log-scale FLOPs plot
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
                all_curves.append(agg)

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
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--std-alpha", type=float, default=0.15)
    args = parser.parse_args()

    set_publication_style()

    entity  = "carlo-romeo-alt427"
    project = "SPEQ"
    dataset = "expert"

    algorithms = [
        "rlpd",
        "calql",
        "speq_o2o",
        "sacfd",
        "faspeq_pct10_pi",
    ]

    envs = ["HalfCheetah-v5"]

    # Per-algorithm seed sets:
    #   baselines → single seed 696969
    #   faspeq    → {696969, 42, 0} with mean ± std
    baseline_seeds = {696969}
    faspeq_seeds   = {696969, 42, 0}

    seeds_per_algo = {
        'rlpd':            baseline_seeds,
        'calql':           baseline_seeds,
        'speq_o2o':        baseline_seeds,
        'sacfd':           baseline_seeds,
        'faspeq_pct10_pi': faspeq_seeds,
    }

    os.makedirs("Plots", exist_ok=True)
    api = wandb.Api()

    print(f"Fetching _flops runs for HalfCheetah-v5 expert...")
    flops_data = fetch_flops_curves(
        api, entity, project, envs, algorithms, seeds_per_algo, dataset
    )

    # 1. Summary table
    print_flops_table(flops_data, algorithms, envs)

    # 2. Log-scale plot
    fig = plot_flops_log(flops_data, algorithms, envs, dataset,
                         args.ema_alpha, args.std_alpha)
    plt.savefig("Plots/flops_halfcheetah_expert.pdf", bbox_inches="tight")
    plt.savefig("Plots/flops_halfcheetah_expert.png", bbox_inches="tight")
    plt.close()
    print("\nSaved: Plots/flops_halfcheetah_expert.{pdf,png}")

    print("\nDONE")
