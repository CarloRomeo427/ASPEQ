import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ─────────────────────────────────────────────────────────────────────────────
# Reusing your existing helper functions
# ─────────────────────────────────────────────────────────────────────────────

def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def get_metric_for_run(api, entity, project, run_name_patterns, metric_name, valid_seeds=None):
    """Generic version of get_eval_reward_for_run to fetch any metric."""
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
            
            # Fetch the specific metric
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
    # Simplified builder for faspeq_pct10_pi
    return f"{algo}_{env}_{dataset}"

def get_env_variants(env):
    variants = [env]
    if env.capitalize() not in variants: variants.append(env.capitalize())
    if env.lower() not in variants: variants.append(env.lower())
    return variants

# ─────────────────────────────────────────────────────────────────────────────
# New Aggregation Logic
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_curves(curves_list, n_points=300):
    """Averages a list of (steps, mean) tuples into a single curve."""
    if not curves_list:
        return None
        
    max_step = max(c[0].max() for c in curves_list)
    common_steps = np.linspace(0, max_step, n_points)
    
    interpolated = []
    for steps, mean in curves_list:
        interpolated.append(np.interp(common_steps, steps, mean))
    
    stacked = np.array(interpolated)
    agg_mean = np.mean(stacked, axis=0)
    agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
    
    return common_steps, agg_mean, agg_std

# ─────────────────────────────────────────────────────────────────────────────
# Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_dataset_envs(dataset_name, env_data_map, metric_name, output_dir):
    """
    Figure 1 Idea: One plot per dataset. 
    Curves: Individual Environments (Ant, Hopper, etc.)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    
    i = 0
    for env, data in env_data_map.items():
        if data is None: continue
        steps, mean, std, _ = data
        
        # Smoothing
        mean_ema = exponential_moving_average(mean, 0.05)
        std_ema = exponential_moving_average(std, 0.05)
        
        color = cmap(i % 10)
        ax.plot(steps, mean_ema, label=env.replace("-v5",""), color=color, linewidth=2)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, color=color, alpha=0.1)
        i += 1
        
    ax.set_title(f"{metric_name} - {dataset_name.upper()} Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Detail_{dataset_name}.png", dpi=300)
    plt.close()

def plot_aggregated_datasets(dataset_agg_map, metric_name, output_dir):
    """
    Figure 2 Idea: One plot comparing Datasets.
    Curves: Expert vs Medium vs Simple (each is an avg of its envs).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'expert': 'red', 'medium': 'blue', 'simple': 'green'}
    
    for dataset, data in dataset_agg_map.items():
        if data is None: continue
        steps, mean, std = data
        
        mean_ema = exponential_moving_average(mean, 0.05)
        std_ema = exponential_moving_average(std, 0.05)
        
        c = colors.get(dataset, 'black')
        ax.plot(steps, mean_ema, label=f"{dataset.upper()}", color=c, linewidth=2.5)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, color=c, alpha=0.1)

    ax.set_title(f"Aggregated {metric_name} per Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Aggregated_Datasets.png", dpi=300)
    plt.close()

def plot_grand_average(grand_agg, metric_name, output_dir):
    """
    Figure 3 Idea: One single curve representing the global average.
    """
    if grand_agg is None: return

    fig, ax = plt.subplots(figsize=(10, 6))
    steps, mean, std = grand_agg
    
    mean_ema = exponential_moving_average(mean, 0.05)
    std_ema = exponential_moving_average(std, 0.05)
    
    ax.plot(steps, mean_ema, label="Grand Average", color='purple', linewidth=3)
    ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, color='purple', alpha=0.15)
    
    ax.set_title(f"Grand Average {metric_name} (All Datasets & Envs)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OfflineEpochs_Grand_Average.png", dpi=300)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str, default="carlo-romeo-alt427")
    parser.add_argument("--wandb-project", type=str, default="SPEQ")
    args = parser.parse_args()

    # Configuration
    ALGO = "faspeq_pct10_pi"
    METRIC = "OfflineEpochs"
    VALID_SEEDS = {0, 42, 1234, 5678, 9876}
    
    DATASETS = ["expert", "medium", "simple"]
    
    # Environments to look for
    ALL_ENVS = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    SIMPLE_ENVS = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

    api = wandb.Api()
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching {METRIC} for {ALGO}...")

    # Data Storage
    # structure: db[dataset][env] = (steps, mean, std, n_seeds)
    db = {ds: {} for ds in DATASETS}

    # 1. Fetch Data
    for dataset in DATASETS:
        target_envs = SIMPLE_ENVS if dataset == "simple" else ALL_ENVS
        print(f"\nProcessing Dataset: {dataset.upper()}")
        
        for env in target_envs:
            # Construct run name variants (e.g. Ant-v5 vs ant-v5)
            env_variants = get_env_variants(env)
            patterns = [build_run_name(ALGO, v, dataset) for v in env_variants]
            
            print(f"  Fetching {env}...", end="", flush=True)
            seeds_data = get_metric_for_run(api, args.wandb_entity, args.wandb_project, patterns, METRIC, VALID_SEEDS)
            
            if seeds_data:
                steps, mean, std = compute_mean_std_across_seeds(seeds_data, METRIC)
                db[dataset][env] = (steps, mean, std, len(seeds_data))
                print(f" Done ({len(seeds_data)} seeds)")
            else:
                db[dataset][env] = None
                print(" No data")

    # 2. Plot Type 1: Combined per dataset (Curves = Envs)
    # 3. Prepare data for Type 2 (Curves = Datasets)
    
    dataset_aggregated_curves = {} # dataset -> (steps, mean, std)

    for dataset in DATASETS:
        # Plot detailed figure
        plot_per_dataset_envs(dataset, db[dataset], METRIC, output_dir)
        
        # Collect valid curves for aggregation
        valid_env_curves = []
        for env, data in db[dataset].items():
            if data is not None:
                # data is (steps, mean, std, n), we just need (steps, mean) for aggregation
                valid_env_curves.append((data[0], data[1]))
        
        if valid_env_curves:
            agg_res = aggregate_curves(valid_env_curves)
            dataset_aggregated_curves[dataset] = agg_res
        else:
            dataset_aggregated_curves[dataset] = None

    # 4. Plot Type 2: Aggregated per dataset (Curves = Datasets)
    plot_aggregated_datasets(dataset_aggregated_curves, METRIC, output_dir)

    # 5. Prepare data for Type 3 (Grand Average)
    all_dataset_curves = []
    for ds, data in dataset_aggregated_curves.items():
        if data is not None:
            all_dataset_curves.append((data[0], data[1]))

    grand_agg = aggregate_curves(all_dataset_curves)

    # 6. Plot Type 3: Grand Average
    plot_grand_average(grand_agg, METRIC, output_dir)

    print(f"\nDone! Plots saved in {output_dir}/")