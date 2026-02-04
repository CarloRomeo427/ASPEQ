import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def auto_label(algo):
    """Auto-generate a clean label from algorithm name."""
    direct = {
        'sacfd': 'SACfD',
        'faspeq_pct10_pi': 'FASPEQ 10% (π)',
        'faspeq_pct10_td': 'FASPEQ 10% (TD)',
    }
    if algo in direct:
        return direct[algo]
    return algo.upper()

def get_eval_reward_for_run(api, entity, project, run_name_patterns, metric_name="EvalReward", valid_seeds=None):
    """Fetch eval reward data for runs matching any of the given patterns."""
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
            
            history = run.history(keys=[metric_name, "_step"], pandas=True)
            if metric_name in history.columns:
                history = history.dropna(subset=[metric_name])
                if len(history) > 0:
                    all_seeds_data.append(history[[metric_name, "_step"]])
                    print(f"    Loaded {run.name} (seed={run.config.get('seed')}): {len(history)} points")
    
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
    """Generate possible capitalization variants for environment names."""
    variants = [env]
    cap_version = env.capitalize()
    if cap_version not in variants:
        variants.append(cap_version)
    lower_version = env.lower()
    if lower_version not in variants:
        variants.append(lower_version)
    if 'Halfcheetah' in env:
        variants.append(env.replace('Halfcheetah', 'HalfCheetah'))
    if 'HalfCheetah' in env:
        variants.append(env.replace('HalfCheetah', 'Halfcheetah'))
    return variants

def build_run_name_variants(algo, env, dataset):
    """
    Build all possible run name patterns.
    
    Naming conventions:
    - sacfd: sacfd_{Env}_{dataset}
    - faspeq_pct10_pi: faspeq_pct10_pi_{Env}_{dataset}
    - faspeq_pct10_td: faspeq_pct10_td_{Env}_{dataset}
    """
    variants = []
    
    for env_variant in get_env_variants(env):
        run_name = f"{algo}_{env_variant}_{dataset}"
        variants.append(run_name)
    
    return variants

def collect_all_data(api, entity, project, envs, algorithms, valid_seeds, dataset, metric_name="EvalReward"):
    """
    Collect all data for all algorithms and environments.
    Returns a nested dict: algo_env_data[algo][env] = (steps, mean, std, n_seeds)
    """
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

def plot_single_env(algo_env_data, env, algorithms, ema_alpha=0.05, std_alpha=0.05, ax=None):
    """Plot a single environment using pre-collected data."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.cm.tab10
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    label_map = {algo: auto_label(algo) for algo in algorithms}
    
    for algo in algorithms:
        data = algo_env_data[algo].get(env)
        if data is None:
            continue
        
        steps, mean, std, n_seeds = data
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        color = color_map[algo]
        label = label_map[algo]
        
        ax.plot(steps, mean_ema, label=label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color)
    
    env_display = env.replace("-v5", "")
    ax.set_title(f"{env_display}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_ylabel("Eval Reward", fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

def compute_aggregated_performance(algo_env_data, algorithms, envs, n_points=300):
    """
    Compute aggregated performance across all environments for each algorithm.
    Returns: dict[algo] = (common_steps, aggregated_mean, aggregated_std, n_envs)
    """
    aggregated = {}
    
    for algo in algorithms:
        env_curves = []
        for env in envs:
            data = algo_env_data[algo].get(env)
            if data is not None:
                steps, mean, std, n_seeds = data
                env_curves.append((steps, mean))
        
        if not env_curves:
            aggregated[algo] = None
            continue
        
        max_step = max(curve[0].max() for curve in env_curves)
        common_steps = np.linspace(0, max_step, n_points)
        
        interpolated_means = []
        for steps, mean in env_curves:
            interp_mean = np.interp(common_steps, steps, mean)
            interpolated_means.append(interp_mean)
        
        stacked = np.array(interpolated_means)
        agg_mean = np.mean(stacked, axis=0)
        agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
        
        aggregated[algo] = (common_steps, agg_mean, agg_std, len(env_curves))
    
    return aggregated

def plot_aggregated(aggregated_data, algorithms, dataset, ema_alpha=0.05, std_alpha=0.1, ax=None):
    """Plot aggregated performance across all environments."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    cmap = plt.cm.tab10
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    label_map = {algo: auto_label(algo) for algo in algorithms}
    
    for algo in algorithms:
        data = aggregated_data.get(algo)
        if data is None:
            continue
        
        steps, mean, std, n_envs = data
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        color = color_map[algo]
        label = f"{label_map[algo]} ({n_envs} envs)"
        
        ax.plot(steps, mean_ema, label=label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color)
    
    ax.set_title(f"Aggregated Performance - {dataset.upper()} Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Eval Reward (averaged across environments)", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

def plot_combined_envs(algo_env_data, envs, algorithms, dataset, ema_alpha, std_alpha):
    """Plot all environments in a single figure."""
    n_envs = len(envs)
    n_cols = min(5, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, env in enumerate(envs):
        plot_single_env(algo_env_data, env, algorithms, ema_alpha, std_alpha, ax=axes[idx])
    
    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"FASPEQ TD vs π Ablation - {dataset.upper()} Dataset", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def print_summary_statistics(algo_env_data, algorithms, envs):
    """Print summary statistics from collected data."""
    label_map = {algo: auto_label(algo) for algo in algorithms}
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Final Scores)")
    print("="*80)
    
    for algo in algorithms:
        final_scores = []
        for env in envs:
            data = algo_env_data[algo].get(env)
            if data is not None:
                steps, mean, std, n_seeds = data
                final_scores.append(mean[-1])
        
        label = label_map[algo]
        if final_scores:
            print(f"{label:30s}: {np.mean(final_scores):8.2f} ± {np.std(final_scores, ddof=1) if len(final_scores) > 1 else 0:6.2f}  (n={len(final_scores)} envs)")
        else:
            print(f"{label:30s}: No data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=["expert", "medium", "simple"],
                        help="Single dataset to plot. If not specified, plots all 3 datasets.")
    parser.add_argument("--individual", action="store_true", 
                        help="Plot individual environment figures instead of combined")
    parser.add_argument("--no-aggregated", action="store_true",
                        help="Skip aggregated performance plots")
    parser.add_argument("--ema-alpha", type=float, default=0.05, 
                        help="EMA smoothing (lower=smoother, 0.01-0.1)")
    parser.add_argument("--std-alpha", type=float, default=0.05, 
                        help="Std shading opacity (0.02-0.1)")
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}
    
    # ============================================================
    # ALGORITHMS TO COMPARE:
    # - sacfd (baseline)
    # - faspeq_pct10_pi (policy loss monitoring)
    # - faspeq_pct10_td (TD error monitoring)
    # ============================================================
    algorithms = [
        "sacfd",
        "faspeq_pct10_pi",
        "faspeq_pct10_td",
    ]
    
    # Environments
    envs = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5", "Ant-v5"]
    
    # Determine which datasets to plot
    if args.dataset:
        datasets_to_plot = [args.dataset]
    else:
        datasets_to_plot = ["expert", "medium", "simple"]
    
    api = wandb.Api()
    
    print("="*80)
    print("FASPEQ TD vs π ABLATION STUDY")
    print("="*80)
    print("ALGORITHMS TO PLOT:")
    for algo in algorithms:
        print(f"  {algo} -> {auto_label(algo)}")
    print("="*80)
    
    # Store all collected data for final aggregated plot across datasets
    all_datasets_data = {}
    
    for dataset in datasets_to_plot:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"Environments: {len(envs)}")
        print(f"{'='*80}")
        
        # Collect all data for this dataset
        algo_env_data = collect_all_data(api, entity, project, envs, algorithms, valid_seeds, dataset)
        all_datasets_data[dataset] = (algo_env_data, envs)
        
        # Plot combined or individual environments
        if args.individual:
            for env in envs:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_single_env(algo_env_data, env, algorithms, args.ema_alpha, args.std_alpha, ax=ax)
                plt.tight_layout()
                plt.savefig(f"Plots/td_vs_pi_{env}_{dataset}.png", dpi=300, bbox_inches="tight")
                plt.close()
            print(f"\nSaved individual plots for {dataset}")
        else:
            fig = plot_combined_envs(algo_env_data, envs, algorithms, dataset, args.ema_alpha, args.std_alpha)
            plt.savefig(f"Plots/td_vs_pi_combined_{dataset}.png", dpi=300, bbox_inches="tight")
            print(f"\nSaved: Plots/td_vs_pi_combined_{dataset}.png")
            plt.close()
        
        # Plot aggregated performance for this dataset
        if not args.no_aggregated:
            aggregated = compute_aggregated_performance(algo_env_data, algorithms, envs)
            fig, ax = plt.subplots(figsize=(12, 7))
            plot_aggregated(aggregated, algorithms, dataset, args.ema_alpha, args.std_alpha * 2, ax=ax)
            plt.tight_layout()
            plt.savefig(f"Plots/td_vs_pi_aggregated_{dataset}.png", dpi=300, bbox_inches="tight")
            print(f"Saved: Plots/td_vs_pi_aggregated_{dataset}.png")
            plt.close()
        
        # Print summary statistics
        print_summary_statistics(algo_env_data, algorithms, envs)
    
    # If plotting all datasets, also create a combined aggregated plot
    if len(datasets_to_plot) == 3 and not args.no_aggregated:
        print(f"\n{'='*80}")
        print("COMPUTING AGGREGATED ACROSS ALL DATASETS")
        print(f"{'='*80}")
        
        # Combine aggregated data across all datasets
        combined_aggregated = {}
        for algo in algorithms:
            all_curves = []
            
            for dataset in datasets_to_plot:
                algo_env_data, dataset_envs = all_datasets_data[dataset]
                
                for env in dataset_envs:
                    data = algo_env_data[algo].get(env)
                    if data is not None:
                        steps, mean, std, n_seeds = data
                        all_curves.append((steps, mean))
            
            if not all_curves:
                combined_aggregated[algo] = None
                continue
            
            max_step = max(curve[0].max() for curve in all_curves)
            common_steps = np.linspace(0, max_step, 300)
            
            interpolated_means = []
            for steps, mean in all_curves:
                interp_mean = np.interp(common_steps, steps, mean)
                interpolated_means.append(interp_mean)
            
            stacked = np.array(interpolated_means)
            agg_mean = np.mean(stacked, axis=0)
            agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
            
            combined_aggregated[algo] = (common_steps, agg_mean, agg_std, len(all_curves))
            print(f"  {auto_label(algo)}: {len(all_curves)} env-dataset pairs")
        
        # Plot combined aggregated
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cmap = plt.cm.tab10
        color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
        label_map = {algo: auto_label(algo) for algo in algorithms}
        
        for algo in algorithms:
            data = combined_aggregated.get(algo)
            if data is None:
                continue
            
            steps, mean, std, n_curves = data
            mean_ema = exponential_moving_average(mean, alpha=args.ema_alpha)
            std_ema = exponential_moving_average(std, alpha=args.ema_alpha)
            
            color = color_map[algo]
            label = f"{label_map[algo]} ({n_curves})"
            
            ax.plot(steps, mean_ema, label=label, linewidth=2, color=color)
            ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.1, color=color)
        
        ax.set_title("FASPEQ TD vs π - ALL DATASETS (Expert + Medium + Simple)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Eval Reward (averaged across all env-dataset pairs)", fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Plots/td_vs_pi_aggregated_all_datasets.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved: Plots/td_vs_pi_aggregated_all_datasets.png")
        plt.close()
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
