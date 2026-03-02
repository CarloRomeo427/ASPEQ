import wandb
import json
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
    """
    Sets a professional, cleaner look for matplotlib figures with increased sizes.
    """
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
        'legend.fontsize': 24,        
        'legend.title_fontsize': 24,  
        'lines.linewidth': 4.5,       
        'lines.markersize': 10,
        'axes.grid': True,
        'grid.alpha': 0.4,            
        'grid.linestyle': '-',        
        'grid.linewidth': 2.5,        # REQUIREMENT: Thicker Grid
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'figure.constrained_layout.use': False,
        'savefig.dpi': 800            
    })

def get_palette(algorithms):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}
    return color_map

def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def auto_label(algo):
    if "faspeq" in algo.lower():
        return "OURS"

    direct = {
        'rlpd': 'RLPD',
        'iql': 'IQL',
        'calql': 'Cal-QL',
        'sacfd': 'SACfD',
        'sac': 'SAC',
        'speq_o2o': 'SPEQ', 
    }
    
    if algo in direct:
        return direct[algo]
    
    label = algo
    if "_o2o" in label:
        return "SPEQ" 
    
    return label.upper()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def get_eval_reward_for_run(api, entity, project, run_name_patterns, metric_name="EvalReward", valid_seeds=None):
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
                except:
                    continue
            try:
                history = run.history(keys=[metric_name, "_step"], pandas=True)
                if metric_name in history.columns:
                    history = history.dropna(subset=[metric_name])
                    if len(history) > 0:
                        all_seeds_data.append(history[[metric_name, "_step"]])
                        print(f"    Loaded {run.name} (seed={run.config.get('seed')}): {len(history)} points")
            except Exception as e:
                print(f"    Error loading {run.name}: {e}")
    return all_seeds_data

def compute_mean_std_across_seeds(all_seeds_data, metric_name="EvalReward"):
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

def get_env_variants(env):
    variants = [env]
    cap_version = env.capitalize()
    if cap_version not in variants:
        variants.append(cap_version)
    lower_version = env.lower()
    if lower_version not in variants:
        variants.append(lower_version)
    return variants

def parse_algo_name(algo):
    if '_valpat' in algo:
        parts = algo.rsplit('_valpat', 1)
        return parts[0], f"valpat{parts[1]}"
    return algo, None

def build_run_name_variants(algo, env, dataset):
    base_algo, suffix = parse_algo_name(algo)
    if base_algo == 'rlpd' and dataset == "expert":
        run_name = f"{base_algo}_{env}"
    else:
        run_name = f"{base_algo}_{env}_{dataset}"
    if suffix:
        run_name = f"{run_name}_{suffix}"
    
    variants = []
    for env_variant in get_env_variants(env):
        variants.append(run_name.replace(env, env_variant))
    return variants

def collect_all_data(api, entity, project, envs, algorithms, valid_seeds, dataset, metric_name="EvalReward"):
    algo_env_data = {algo: {} for algo in algorithms}
    for env in envs:
        print(f"\n{env}:")
        for algo in algorithms:
            run_patterns = build_run_name_variants(algo, env, dataset)
            print(f"  Looking for {algo}: {run_patterns[0]}...")
            all_seeds_data = get_eval_reward_for_run(api, entity, project, run_patterns, metric_name, valid_seeds)
            if not all_seeds_data:
                print(f"    No data found")
                algo_env_data[algo][env] = None
                continue
            steps, mean, std = compute_mean_std_across_seeds(all_seeds_data, metric_name)
            algo_env_data[algo][env] = (steps, mean, std, len(all_seeds_data))
            print(f"    Found {len(all_seeds_data)} seeds, final: {mean[-1]:.2f} ± {std[-1]:.2f}")
    return algo_env_data

# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregation & Normalization
# ─────────────────────────────────────────────────────────────────────────────

def load_normalizing_boundaries(path="normalizing_boundaries.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        return json.load(f)

def normalize_score(raw_score, env_min, env_max):
    denom = env_max - env_min
    if abs(denom) < 1e-8:
        if isinstance(raw_score, np.ndarray):
            return np.zeros_like(raw_score)
        return 0.0
    return (raw_score - env_min) / denom

def compute_aggregated_performance(algo_env_data, algorithms, envs, boundaries=None, n_points=300):
    aggregated = {}
    for algo in algorithms:
        env_curves = []
        for env in envs:
            data = algo_env_data[algo].get(env)
            if data is None:
                continue
            steps, mean, std, n_seeds = data
            
            if boundaries and env in boundaries:
                env_min = boundaries[env]["min"]
                env_max = boundaries[env]["max"]
                mean = normalize_score(mean, env_min, env_max)
                std = std / (env_max - env_min) if abs(env_max - env_min) > 1e-8 else std
            
            env_curves.append((steps, mean))
            
        if not env_curves:
            aggregated[algo] = None
            continue
            
        max_step = max(c[0].max() for c in env_curves)
        common_steps = np.linspace(0, max_step, n_points)
        interpolated = [np.interp(common_steps, steps, mean) for steps, mean in env_curves]
        stacked = np.array(interpolated)
        
        agg_mean = np.mean(stacked, axis=0)
        agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
        aggregated[algo] = (common_steps, agg_mean, agg_std, len(env_curves))
    return aggregated

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_ax(ax, algo_env_data, env, algorithms, boundaries, ema_alpha, std_alpha, color_map, label_map):
    """Helper to plot one environment onto a given axes object."""
    
    for algo in algorithms:
        data = algo_env_data[algo].get(env)
        if data is None:
            continue
        
        steps, mean, std, n_seeds = data
        
        if boundaries and env in boundaries:
            b = boundaries[env]
            mean = normalize_score(mean, b["min"], b["max"])
            std = std / (b["max"] - b["min"]) if abs(b["max"] - b["min"]) > 1e-8 else std

        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        color = color_map[algo]
        label = label_map[algo] 
        
        ax.plot(steps, mean_ema, label=label, linewidth=4.5, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color, linewidth=0)
        
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)

def plot_combined_pretty(algo_env_data, algorithms, dataset, boundaries, ema_alpha, std_alpha):
    """
    Plots a grid with a single horizontal legend in a box at the top.
    """
    main_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    side_envs = ["InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    
    # REQUIREMENT 2: Eliminate bottom row for "simple" dataset
    if dataset == "simple":
        # 1 Row only
        fig, axes = plt.subplots(1, 5, figsize=(30, 8), sharey=False) 
        # Make iterable consistent for loop
        if not isinstance(axes, np.ndarray): axes = np.array([axes])
        rows = [axes] 
        env_sets = [main_envs]
        x_label_row_idx = 0 
    else:
        # 2 Rows
        fig, axes = plt.subplots(2, 5, figsize=(30, 14), sharey=False)
        rows = [axes[0], axes[1]]
        env_sets = [main_envs, side_envs]
        x_label_row_idx = 1 

    color_map = get_palette(algorithms)
    label_map = {algo: auto_label(algo) for algo in algorithms}

    # Iterate through rows/env_sets
    for r_idx, (row_axes, current_envs) in enumerate(zip(rows, env_sets)):
        for i, env in enumerate(current_envs):
            ax = row_axes[i]
            plot_single_ax(ax, algo_env_data, env, algorithms, boundaries, ema_alpha, std_alpha, color_map, label_map)
            
            # Y Label on first column only
            if i == 0: 
                ax.set_ylabel("Normalized Score", fontsize=22, labelpad=12)
            
            # X Label on bottom row (or only row)
            if r_idx == x_label_row_idx:
                ax.set_xlabel("Steps", fontsize=22, labelpad=12)
            
            # Env Name annotation
            ax.text(0.5, 1.02, env.replace("-v5",""), transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=24, fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────────
    # REQUIREMENT 1 & 4: Horizontal Legend in a Box
    # ─────────────────────────────────────────────────────────────────────────
    
    # Manually create handles to ensure legend exists even if data is missing in first plot
    legend_handles = [Line2D([0], [0], color=color_map[algo], lw=4.5) for algo in algorithms]
    legend_labels = [label_map[algo] for algo in algorithms]

    # Adjust layout reservation based on single or double row
    if dataset == "simple":
        # Single row need less vertical height, legend goes above
        rect_param = [0, 0, 1, 0.85]  # Reserve top 15% for legend
        legend_y = 0.95
    else:
        # Double row
        rect_param = [0, 0, 1, 0.90]  # Reserve top 10% for legend
        legend_y = 0.96

    # Create the legend on the FIGURE object (not axes)
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='upper center',            # Anchor point on the legend box
        bbox_to_anchor=(0.5, legend_y),# Position relative to figure (0,0 to 1,1)
        ncol=len(algorithms),          # Horizontal
        fontsize=24, 
        frameon=True,                  # Box ON
        framealpha=1.0,                # Opaque
        edgecolor='black',             # Black border
        fancybox=False,                # Square corners
        columnspacing=1.5
    )

    plt.tight_layout(rect=rect_param) 
    return fig

def plot_aggregated_pretty(aggregated_data, algorithms, ema_alpha=0.05, std_alpha=0.1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
    color_map = get_palette(algorithms)
    label_map = {algo: auto_label(algo) for algo in algorithms}
    
    for algo in algorithms:
        data = aggregated_data.get(algo)
        if data is None:
            continue
            
        steps, mean, std, count = data
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        color = color_map[algo]
        label = label_map[algo]

        ax.plot(steps, mean_ema, label=label, linewidth=4.5, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color, linewidth=0)
    
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    ax.set_xlabel("Steps", fontsize=22)
    ax.set_ylabel("Normalized Score", fontsize=22)
    ax.grid(True, linestyle='-', linewidth=2.5, alpha=0.4)
    
    # Legend in box for aggregated plot too
    ax.legend(fontsize=20, loc='lower right', frameon=True, framealpha=1.0, edgecolor='black', fancybox=False)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Table Logic
# ─────────────────────────────────────────────────────────────────────────────

def collect_table_scores(all_datasets_data, algorithms, boundaries):
    scores = {}
    dataset_averages = {}
    for dataset, (algo_env_data, envs) in all_datasets_data.items():
        scores[dataset] = {}
        dataset_averages[dataset] = {algo: [] for algo in algorithms}
        for env in envs:
            scores[dataset][env] = {}
            for algo in algorithms:
                data = algo_env_data[algo].get(env)
                if data is None:
                    scores[dataset][env][algo] = None
                    continue
                steps, mean, std, n_seeds = data
                window = min(10, len(mean))
                final_mean = np.mean(mean[-window:])
                final_std = np.mean(std[-window:])
                if boundaries and env in boundaries:
                    env_min = boundaries[env]["min"]
                    env_max = boundaries[env]["max"]
                    denom = env_max - env_min
                    final_mean = normalize_score(final_mean, env_min, env_max)
                    final_std = final_std / denom if abs(denom) > 1e-8 else final_std
                scores[dataset][env][algo] = (final_mean, final_std, n_seeds)
                dataset_averages[dataset][algo].append(final_mean)
    return scores, dataset_averages

def generate_summary_table(dataset_averages, algorithms):
    summary = {} 
    for dataset, algo_vals in dataset_averages.items():
        row_name = f"{dataset.capitalize()} Average"
        summary[row_name] = {}
        for algo in algorithms:
            vals = algo_vals[algo]
            summary[row_name][algo] = np.mean(vals) if vals else None
    summary["Total Average"] = {}
    for algo in algorithms:
        means = []
        for dataset in dataset_averages:
            m = summary[f"{dataset.capitalize()} Average"].get(algo)
            if m is not None: means.append(m)
        summary["Total Average"][algo] = np.mean(means) if means else None
    return summary

def format_cell(mean, std=None, is_best=False, fmt=".1f"):
    if mean is None: return "—"
    text = f"{mean:{fmt}} \\pm {std:{fmt}}" if std is not None else f"{mean:{fmt}}"
    return f"$\\mathbf{{{text}}}$" if is_best else f"${text}$"

def generate_latex_complete(scores, summary_data, algorithms, our_algo):
    label_map = {a: auto_label(a) for a in algorithms}
    latex_labels = {a: (f"\\textbf{{{label_map[a]}}}" if "OURS" in label_map[a] else label_map[a]) for a in algorithms}
    
    all_latex = []
    
    for dataset, env_dict in scores.items():
        envs = list(env_dict.keys())
        lines = []
        lines.append(f"\\begin{{table*}}[ht]")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{Normalized scores (\\%) on \\textbf{{{dataset}}} dataset.}}")
        lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{")
        lines.append(f"\\begin{{tabular}}{{l" + "c" * len(algorithms) + "}")
        lines.append(f"\\toprule")
        lines.append("Environment & " + " & ".join(latex_labels[a] for a in algorithms) + " \\\\")
        lines.append("\\midrule")
        for env in envs:
            env_display = env.replace("-v5", "")
            means = {a: env_dict[env][a][0] for a in algorithms if env_dict[env][a]}
            best_algo = max(means, key=means.get) if means else None
            cells = [format_cell(env_dict[env].get(a)[0]*100, env_dict[env].get(a)[1]*100, is_best=(a==best_algo)) if env_dict[env].get(a) else "—" for a in algorithms]
            lines.append(f"{env_display} & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")
        lines.append("\\end{table*}")
        all_latex.append("\n".join(lines))
        
    lines = []
    lines.append(f"\\begin{{table*}}[ht]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{Summary of Normalized Scores (Average across environments).}}")
    lines.append(f"\\begin{{tabular}}{{l" + "c" * len(algorithms) + "}")
    lines.append(f"\\toprule")
    lines.append("Dataset & " + " & ".join(latex_labels[a] for a in algorithms) + " \\\\")
    lines.append("\\midrule")
    row_order = [k for k in summary_data.keys() if "Total" not in k] + ["Total Average"]
    for row_name in row_order:
        means = {a: summary_data[row_name][a] for a in algorithms if summary_data[row_name][a] is not None}
        best_algo = max(means, key=means.get) if means else None
        cells = [format_cell(summary_data[row_name].get(a)*100, is_best=(a==best_algo)) if summary_data[row_name].get(a) is not None else "—" for a in algorithms]
        if row_name == "Total Average":
            lines.append("\\midrule")
            lines.append(f"\\textbf{{{row_name}}} & " + " & ".join(cells) + " \\\\")
        else:
             lines.append(f"{row_name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    all_latex.append("\n".join(lines))
    return "\n\n".join(all_latex)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=["expert", "medium", "simple"])
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--std-alpha", type=float, default=0.15)
    parser.add_argument("--boundaries", type=str, default="normalizing_boundaries.json")
    parser.add_argument("--our-algo", type=str, default="faspeq_pct10_pi")
    args = parser.parse_args()

    set_publication_style()

    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}
    algorithms = ["speq_o2o", "rlpd", "calql", "sacfd", "faspeq_pct10_pi"]
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    
    datasets_to_process = [args.dataset] if args.dataset else ["expert", "medium", "simple"]

    try:
        boundaries = load_normalizing_boundaries(args.boundaries)
    except FileNotFoundError:
        print(f"ERROR: {args.boundaries} not found.")
        exit(1)

    api = wandb.Api()
    os.makedirs("Plots", exist_ok=True)
    
    all_datasets_data = {}
    aggregated_per_dataset = {} 

    # 1. Process Datasets
    for dataset in datasets_to_process:
        print(f"\nProcessing DATASET: {dataset.upper()}")
        algo_env_data = collect_all_data(api, entity, project, all_envs, algorithms, valid_seeds, dataset)
        all_datasets_data[dataset] = (algo_env_data, all_envs)

        # Plot 1: Combined Pretty
        fig = plot_combined_pretty(algo_env_data, algorithms, dataset, boundaries, args.ema_alpha, args.std_alpha)
        plt.savefig(f"Plots/combined_split_{dataset}.pdf") 
        plt.savefig(f"Plots/combined_split_{dataset}.png") 
        plt.close()
        
        # Plot 2: Aggregated (Per Dataset)
        agg_data = compute_aggregated_performance(algo_env_data, algorithms, all_envs, boundaries)
        aggregated_per_dataset[dataset] = agg_data
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_aggregated_pretty(agg_data, algorithms, args.ema_alpha, args.std_alpha, ax=ax)
        plt.tight_layout()
        plt.savefig(f"Plots/aggregated_{dataset}.pdf")
        plt.savefig(f"Plots/aggregated_{dataset}.png")
        plt.close()

    # 2. Grand Average Plot
    if len(aggregated_per_dataset) > 1:
        print("\nComputing Grand Average...")
        grand_aggregated = {}
        for algo in algorithms:
            dataset_means = []
            steps_ref = None
            for dataset in aggregated_per_dataset:
                data = aggregated_per_dataset[dataset].get(algo)
                if data is not None:
                    steps, mean, std, _ = data
                    dataset_means.append(mean)
                    if steps_ref is None: steps_ref = steps
            if dataset_means:
                stacked = np.array(dataset_means)
                grand_mean = np.mean(stacked, axis=0)
                grand_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(grand_mean)
                grand_aggregated[algo] = (steps_ref, grand_mean, grand_std, len(dataset_means))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_aggregated_pretty(grand_aggregated, algorithms, args.ema_alpha, args.std_alpha, ax=ax)
        plt.tight_layout()
        plt.savefig(f"Plots/aggregated_grand_average.pdf")
        plt.savefig(f"Plots/aggregated_grand_average.png")
        plt.close()

    # 3. Tables
    scores, dataset_averages = collect_table_scores(all_datasets_data, algorithms, boundaries)
    summary_data = generate_summary_table(dataset_averages, algorithms)
    latex_output = generate_latex_complete(scores, summary_data, algorithms, our_algo=args.our_algo)
    
    with open("Plots/results_table.tex", "w") as f:
        f.write(latex_output)
    print(f"\nLaTeX table saved to: Plots/results_table.tex")
    print("DONE.")