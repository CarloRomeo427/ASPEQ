import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import argparse
import os


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


def get_palette(n):
    """Return n colours from tab10."""
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n)]


# Fixed colours for the dataset-comparison plot
DATASET_COLORS = {
    'expert': '#d62728',   # red
    'medium': '#1f77b4',   # blue
    'simple': '#2ca02c',   # green
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Utilities
# ─────────────────────────────────────────────────────────────────────────────

def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def get_metric_for_run(api, entity, project, run_name_patterns, metric_name, valid_seeds=None):
    if isinstance(run_name_patterns, str):
        run_name_patterns = [run_name_patterns]

    runs = api.runs(f"{entity}/{project}")
    all_seeds_data = []

    for run in runs:
        if run.name in run_name_patterns:
            if valid_seeds is not None:
                try:
                    seed = run.config.get('seed', None)
                    if seed not in valid_seeds:
                        continue
                except Exception:
                    continue

            history = run.history(keys=[metric_name, "_step"], pandas=True)
            if metric_name in history.columns:
                history = history.dropna(subset=[metric_name])
                if len(history) > 0:
                    all_seeds_data.append(history[[metric_name, "_step"]])
                    print(f"    Loaded {run.name} (seed={run.config.get('seed')}): {len(history)} points")

    return all_seeds_data


def compute_mean_std_across_seeds(all_seeds_data, metric_name):
    if not all_seeds_data:
        return None, None, None

    all_steps = set()
    for df in all_seeds_data:
        all_steps.update(df["_step"].values)
    all_steps = sorted(all_steps)

    interpolated_seeds = []
    for df in all_seeds_data:
        df_sorted = df.sort_values("_step")
        interp_values = np.interp(all_steps, df_sorted["_step"].values, df_sorted[metric_name].values)
        interpolated_seeds.append(interp_values)

    stacked = np.array(interpolated_seeds)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)

    return np.array(all_steps), mean, std


def build_run_name(algo, env, dataset):
    return f"{algo}_{env}_{dataset}"


def get_env_variants(env):
    variants = [env]
    if env.capitalize() not in variants:
        variants.append(env.capitalize())
    if env.lower() not in variants:
        variants.append(env.lower())
    return variants


def aggregate_curves(curves_list, n_points=300):
    """Averages a list of (steps, mean) tuples into a single curve."""
    if not curves_list:
        return None

    max_step = max(c[0].max() for c in curves_list)
    common_steps = np.linspace(0, max_step, n_points)

    interpolated = [np.interp(common_steps, steps, mean) for steps, mean in curves_list]
    stacked = np.array(interpolated)
    agg_mean = np.mean(stacked, axis=0)
    agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)

    return common_steps, agg_mean, agg_std


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_dataset_envs(dataset_name, env_data_map, metric_name, output_dir,
                          ema_alpha=0.05, std_alpha=0.15):
    """
    One plot per dataset. Curves = individual environments.
    No title. Legend bottom-right, smaller, framed.
    """
    envs = [e for e, d in env_data_map.items() if d is not None]
    colors = get_palette(len(envs))
    color_map = {env: colors[i] for i, env in enumerate(envs)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for env in envs:
        data = env_data_map[env]
        steps, mean, std, _ = data

        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema  = exponential_moving_average(std,  alpha=ema_alpha)

        color = color_map[env]
        label = env.replace("-v5", "")
        ax.plot(steps, mean_ema, label=label, linewidth=4.5, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema,
                        color=color, alpha=std_alpha, linewidth=0)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel("Steps", fontsize=22, labelpad=12)
    ax.set_ylabel("OfflineUpdates", fontsize=22, labelpad=12)
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)

    # Legend — bottom right, smaller, framed
    legend_handles = [Line2D([0], [0], color=color_map[e], lw=4.5) for e in envs]
    legend_labels  = [e.replace("-v5", "") for e in envs]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        fontsize=13,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
        columnspacing=1.2,
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Detail_{dataset_name}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/OfflineEpochs_Detail_{dataset_name}.png", bbox_inches='tight')
    plt.close()


def plot_aggregated_datasets(dataset_agg_map, metric_name, output_dir,
                             ema_alpha=0.05, std_alpha=0.15):
    """
    One plot comparing datasets. Curves = Expert vs Medium vs Simple.
    No title. Legend bottom-right, smaller, framed.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    datasets = [ds for ds, d in dataset_agg_map.items() if d is not None]

    for dataset in datasets:
        steps, mean, std = dataset_agg_map[dataset]

        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema  = exponential_moving_average(std,  alpha=ema_alpha)

        color = DATASET_COLORS.get(dataset, '#888888')
        ax.plot(steps, mean_ema, label=dataset.upper(), linewidth=4.5, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema,
                        color=color, alpha=std_alpha, linewidth=0)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel("Steps", fontsize=22, labelpad=12)
    ax.set_ylabel("OfflineUpdates", fontsize=22, labelpad=12)
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)

    legend_handles = [Line2D([0], [0], color=DATASET_COLORS.get(ds, '#888888'), lw=4.5)
                      for ds in datasets]
    legend_labels  = [ds.upper() for ds in datasets]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        fontsize=13,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
        columnspacing=1.2,
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Aggregated_Datasets.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/OfflineEpochs_Aggregated_Datasets.png", bbox_inches='tight')
    plt.close()


def plot_grand_average(grand_agg, metric_name, output_dir,
                       ema_alpha=0.05, std_alpha=0.15):
    """
    Single curve: global average across all datasets and environments.
    No title. Legend bottom-right, smaller, framed.
    """
    if grand_agg is None:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    steps, mean, std = grand_agg

    mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
    std_ema  = exponential_moving_average(std,  alpha=ema_alpha)

    color = '#7B2D8B'  # violet (matches OURS colour)
    ax.plot(steps, mean_ema, label="Grand Average", linewidth=4.5, color=color)
    ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema,
                    color=color, alpha=std_alpha, linewidth=0)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel("Steps", fontsize=22, labelpad=12)
    ax.set_ylabel("OfflineUpdates", fontsize=22, labelpad=12)
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)

    ax.legend(
        loc='lower right',
        fontsize=16,
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        fancybox=False,
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Grand_Average.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/OfflineEpochs_Grand_Average.png", bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str, default="carlo-romeo-alt427")
    parser.add_argument("--wandb-project", type=str, default="SPEQ")
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--std-alpha", type=float, default=0.15)
    args = parser.parse_args()

    set_publication_style()

    # Configuration
    ALGO        = "faspeq_pct10_pi"
    METRIC      = "OfflineEpochs"
    VALID_SEEDS = {0, 42, 1234, 5678, 9876}
    DATASETS    = ["expert", "medium", "simple"]

    ALL_ENVS    = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    SIMPLE_ENVS = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

    api        = wandb.Api()
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching {METRIC} for {ALGO}...")

    # structure: db[dataset][env] = (steps, mean, std, n_seeds)
    db = {ds: {} for ds in DATASETS}

    # 1. Fetch Data
    for dataset in DATASETS:
        target_envs = SIMPLE_ENVS if dataset == "simple" else ALL_ENVS
        print(f"\nProcessing Dataset: {dataset.upper()}")

        for env in target_envs:
            env_variants = get_env_variants(env)
            patterns     = [build_run_name(ALGO, v, dataset) for v in env_variants]

            print(f"  Fetching {env}...", end="", flush=True)
            seeds_data = get_metric_for_run(
                api, args.wandb_entity, args.wandb_project,
                patterns, METRIC, VALID_SEEDS
            )

            if seeds_data:
                steps, mean, std = compute_mean_std_across_seeds(seeds_data, METRIC)
                db[dataset][env] = (steps, mean, std, len(seeds_data))
                print(f" Done ({len(seeds_data)} seeds)")
            else:
                db[dataset][env] = None
                print(" No data")

    # 2. Per-dataset detail plots + aggregation
    dataset_aggregated_curves = {}

    for dataset in DATASETS:
        plot_per_dataset_envs(dataset, db[dataset], METRIC, output_dir,
                              args.ema_alpha, args.std_alpha)

        valid_env_curves = [
            (data[0], data[1])
            for data in db[dataset].values()
            if data is not None
        ]

        dataset_aggregated_curves[dataset] = (
            aggregate_curves(valid_env_curves) if valid_env_curves else None
        )

    # 3. Aggregated datasets comparison plot
    plot_aggregated_datasets(dataset_aggregated_curves, METRIC, output_dir,
                             args.ema_alpha, args.std_alpha)

    # 4. Grand average plot
    all_dataset_curves = [
        (data[0], data[1])
        for data in dataset_aggregated_curves.values()
        if data is not None
    ]
    grand_agg = aggregate_curves(all_dataset_curves)
    plot_grand_average(grand_agg, METRIC, output_dir,
                       args.ema_alpha, args.std_alpha)

    print(f"\nDone! Plots saved in {output_dir}/")