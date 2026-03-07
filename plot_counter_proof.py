"""
Plot SPEQ O2O offline_epochs ablation + FASPEQ on HalfCheetah-v5 expert.
Fetches runs by displayName from WandB.
"""

import wandb
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Style (from main plotting script)
# ─────────────────────────────────────────────────────────────────────────────

def set_publication_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'legend.title_fontsize': 20,
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


def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def load_normalizing_boundaries(path="normalizing_boundaries.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        return json.load(f)

def normalize_score(raw_score, env_min, env_max):
    denom = env_max - env_min
    if abs(denom) < 1e-8:
        return np.zeros_like(raw_score) if isinstance(raw_score, np.ndarray) else 0.0
    return (raw_score - env_min) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

ENTITY = "carlo-romeo-alt427"
PROJECT = "SPEQ"
METRIC = "EvalReward"
VALID_SEEDS = {0, 42, 1234, 5678, 9876, 777, 24680, 13579, 31415, 27182}

# Runs to plot: (displayName_pattern, label)
# displayName is set by wandb.init(name=...) in train.py
RUNS = [
    ("speq_o2o_Halfcheetah-v5_expert_10000",  "SPEQ (10k)"),
    ("speq_o2o_Halfcheetah-v5_expert_25000",  "SPEQ (25k)"),
    ("speq_o2o_Halfcheetah-v5_expert_50000",  "SPEQ (50k)"),
    ("speq_o2o_HalfCheetah-v5_expert",         "SPEQ (75k)"),
    ("speq_o2o_Halfcheetah-v5_expert_100000",  "SPEQ (100k)"),
    # ("faspeq_pct10_pi_HalfCheetah-v5_expert",  "OURS"),
]


def fetch_all_runs(api):
    """Fetch all matching runs, grouped by display name pattern."""
    all_project_runs = api.runs(f"{ENTITY}/{PROJECT}")

    # Index runs by LOWERCASED name for case-insensitive matching
    name_to_runs = {}
    for run in all_project_runs:
        name_to_runs.setdefault(run.name.lower(), []).append(run)

    results = {}
    for pattern, label in RUNS:
        matching = name_to_runs.get(pattern.lower(), [])
        
        # Collect per-seed, keeping only the longest run per seed
        best_per_seed = {}

        for run in matching:
            seed = run.config.get("seed", None)
            if seed not in VALID_SEEDS:
                continue
            try:
                history = run.history(keys=[METRIC, "_step"], pandas=True)
                history = history.dropna(subset=[METRIC])
                if len(history) == 0:
                    continue
                # Keep the run with the most data points for each seed
                if seed not in best_per_seed or len(history) > len(best_per_seed[seed]):
                    best_per_seed[seed] = history[[METRIC, "_step"]]
                    print(f"  Loaded {run.name} seed={seed}: {len(history)} pts")
            except Exception as e:
                print(f"  Error {run.name}: {e}")

        seeds_data = list(best_per_seed.values())

        if not seeds_data:
            print(f"  WARNING: No data for '{pattern}'")
            results[label] = None
            continue

        # Interpolate to common steps and compute mean ± std
        all_steps = sorted(set().union(*(df["_step"].values for df in seeds_data)))
        interpolated = []
        for df in seeds_data:
            df_s = df.sort_values("_step")
            interpolated.append(np.interp(all_steps, df_s["_step"].values, df_s[METRIC].values))

        stacked = np.array(interpolated)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
        results[label] = (np.array(all_steps), mean, std, len(seeds_data))
        print(f"  {label}: {len(seeds_data)} seeds, final={mean[-1]:.1f} ± {std[-1]:.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(results, ema_alpha=0.05, std_alpha=0.15, boundaries=None, env="HalfCheetah-v5"):
    set_publication_style()

    fig, ax = plt.subplots(figsize=(14, 9))

    cmap = plt.get_cmap("tab10")
    labels_ordered = [label for _, label in RUNS]
    color_map = {label: cmap(i) for i, label in enumerate(labels_ordered)}

    for label in labels_ordered:
        data = results.get(label)
        if data is None:
            continue

        steps, mean, std, n_seeds = data

        # Normalize
        if boundaries and env in boundaries:
            b = boundaries[env]
            mean = normalize_score(mean, b["min"], b["max"])
            std = std / (b["max"] - b["min"]) if abs(b["max"] - b["min"]) > 1e-8 else std

        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        color = color_map[label]

        ax.plot(steps, mean_ema, label=f"{label}", linewidth=4.5, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema,
                        alpha=std_alpha, color=color, linewidth=0)

    ax.set_xlabel("Steps", fontsize=22, labelpad=12)
    ax.set_ylabel("Normalized Score", fontsize=22, labelpad=12)
    # ax.set_title("HalfCheetah-v5 Expert — Offline Epochs Ablation", fontsize=24, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="-", linewidth=2.5, alpha=0.4)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    api = wandb.Api()

    try:
        boundaries = load_normalizing_boundaries("normalizing_boundaries.json")
    except FileNotFoundError:
        print("WARNING: normalizing_boundaries.json not found, plotting raw rewards")
        boundaries = None

    print("Fetching runs...")
    results = fetch_all_runs(api)
    fig = plot(results, boundaries=boundaries)
    os.makedirs("Plots", exist_ok=True)
    fig.savefig("Plots/halfcheetah_offline_epochs_ablation.pdf", bbox_inches="tight")
    fig.savefig("Plots/halfcheetah_offline_epochs_ablation.png", bbox_inches="tight")
    print("\nSaved to Plots/halfcheetah_offline_epochs_ablation.{pdf,png}")
    plt.show()