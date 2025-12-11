import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def exponential_moving_average(data, alpha=0.05):
    """Apply EMA smoothing to data."""
    if len(data) == 0:
        return data
    
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def get_eval_reward_for_run(api, entity, project, run_name_pattern, metric_name="EvalReward", valid_seeds=None):
    """Get eval reward data for all seeds of a run."""
    runs = api.runs(f"{entity}/{project}")
    
    all_seeds_data = []
    
    for run in runs:
        if run.name.startswith(run_name_pattern):
            # Filter by seed if valid_seeds is provided
            if valid_seeds is not None:
                try:
                    seed = run.config.get('seed', None)
                    if seed not in valid_seeds:
                        continue
                except:
                    continue
            
            history = run.history(keys=[metric_name, "_step"], pandas=True)
            
            if metric_name in history.columns:
                # Drop NaN values
                history = history.dropna(subset=[metric_name])
                if len(history) > 0:
                    all_seeds_data.append(history[[metric_name, "_step"]])
                    print(f"Loaded {run.name}: {len(history)} points")
    
    return all_seeds_data

def collect_final_scores_and_times(api, entity, project, environments, algorithms, valid_seeds, metric_name="EvalReward"):
    """
    Collect final scores and running times for all algorithms across all environments and seeds.
    
    Returns:
        dict: {algo: {'final_scores': [scores], 'run_times': [times in hours]}}
    """
    runs = api.runs(f"{entity}/{project}")
    
    algo_data = {algo: {'final_scores': [], 'run_times': []} for algo in algorithms}
    
    print("\n" + "="*80)
    print("COLLECTING FINAL SCORES AND RUNNING TIMES")
    print("="*80)
    
    for env in environments:
        print(f"\nProcessing {env}...")
        
        for algo in algorithms:
            run_pattern = f"{algo}_{env}"
            
            for run in runs:
                if run.name.startswith(run_pattern):
                    # Check if seed is valid
                    try:
                        seed = run.config.get('seed', None)
                        if seed not in valid_seeds:
                            continue
                    except:
                        continue
                    
                    # Get final score
                    history = run.history(keys=[metric_name], pandas=True)
                    if metric_name in history.columns:
                        history = history.dropna(subset=[metric_name])
                        if len(history) > 0:
                            final_score = history[metric_name].iloc[-1]
                            algo_data[algo]['final_scores'].append(final_score)
                            
                            # Get running time (in seconds, convert to hours)
                            # WandB stores runtime in summary._wandb.runtime (seconds)
                            try:
                                runtime_seconds = run.summary.get('_wandb', {}).get('runtime', None)
                                if runtime_seconds is not None:
                                    runtime_hours = runtime_seconds / 3600.0
                                    algo_data[algo]['run_times'].append(runtime_hours)
                                    print(f"  {run.name}: Final={final_score:.2f}, Time={runtime_hours:.2f}h")
                                else:
                                    print(f"  {run.name}: Final={final_score:.2f}, Time=N/A")
                            except:
                                print(f"  {run.name}: Final={final_score:.2f}, Time=N/A")
    
    return algo_data

def print_summary_statistics(algo_data, algorithms):
    """Print summary statistics for final scores and running times."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nFINAL SCORES (averaged across all environments and seeds):")
    print("-" * 80)
    for algo in algorithms:
        scores = algo_data[algo]['final_scores']
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0
            print(f"{algo.upper():8s}: {mean_score:8.2f} ± {std_score:6.2f}  (n={len(scores)} runs)")
        else:
            print(f"{algo.upper():8s}: No data")
    
    print("\nRUNNING TIMES (averaged across all environments and seeds):")
    print("-" * 80)
    for algo in algorithms:
        times = algo_data[algo]['run_times']
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times, ddof=1) if len(times) > 1 else 0
            print(f"{algo.upper():8s}: {mean_time:6.2f} ± {std_time:5.2f} hours  (n={len(times)} runs)")
        else:
            print(f"{algo.upper():8s}: No timing data available")
    
    print("="*80)

def compute_mean_std_across_seeds(all_seeds_data, metric_name="EvalReward"):
    """Compute mean and std across seeds, aligned by step."""
    if not all_seeds_data:
        return None, None, None
    
    # Find all unique steps
    all_steps = set()
    for df in all_seeds_data:
        all_steps.update(df["_step"].values)
    all_steps = sorted(all_steps)
    
    # Interpolate each seed to common steps
    interpolated_seeds = []
    for df in all_seeds_data:
        df_sorted = df.sort_values("_step")
        # Interpolate to common steps
        interp_values = np.interp(all_steps, df_sorted["_step"].values, df_sorted[metric_name].values)
        interpolated_seeds.append(interp_values)
    
    # Stack and compute statistics
    stacked = np.array(interpolated_seeds)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
    
    return np.array(all_steps), mean, std

def plot_single_env(api, entity, project, env, algorithms, valid_seeds, metric_name="EvalReward", ema_alpha=0.05, ax=None):
    """Plot a single environment on given axis."""
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Explicit color mapping
    color_map = {
        'sac': '#d62728',      # Red
        'speq': '#2ca02c',     # Green
        'droq': '#9467bd',     # Purple
        'paspeq': '#1f77b4',   # Blue
        'aspeq': '#ff7f0e',    # Orange
    }
    
    for idx, algo in enumerate(algorithms):
        run_pattern = f"{algo}_{env}"
        
        # Get data for valid seeds only
        all_seeds_data = get_eval_reward_for_run(api, entity, project, run_pattern, metric_name, valid_seeds)
        
        if not all_seeds_data:
            print(f"No data found for {run_pattern}")
            continue
        
        # Compute mean and std
        steps, mean, std = compute_mean_std_across_seeds(all_seeds_data, metric_name)
        
        # Apply EMA
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        # Label formatting
        if algo == 'speq':
            algo_label = 'SPEQ (Ours)'
        elif algo == 'paspeq':
            algo_label = 'PASPEQ (Ours)'
        else:
            algo_label = algo.upper()
        
        # Plot with explicit color
        color = color_map.get(algo.lower(), plt.cm.tab10(idx))
        ax.plot(steps, mean_ema, label=algo_label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.2, color=color)
        
        print(f"{algo_label}: {mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
    
    # Format axis
    env_display = env.replace("-v5", "")
    ax.set_title(f"{env_display}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_ylabel("Eval Reward", fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    return True

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    # Hardcoded seed list
    valid_seeds = {0, 42, 1234, 5678, 777, 9876, 13579, 31415, 24680, 27182}
    
    # Main environments (row 1)
    main_envs = ["Ant-v5", "Walker2d-v5", "HalfCheetah-v5", "Humanoid-v5", "Hopper-v5"]
    
    # Other environments (row 2)
    other_envs = ["InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Reacher-v5", "Swimmer-v5", "Pusher-v5"]
    
    # Combine all environments
    all_envs = main_envs  + other_envs
    
    algorithms = ["paspeq", "speq", "droq", "sac"]
    
    # Create combined plot with 2 rows
    api = wandb.Api()
    
    # Create figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    print(f"Using seeds: {sorted(valid_seeds)}\n")
    
    # Plot main environments in first row
    for idx, env in enumerate(main_envs):
        print(f"\n{env}:")
        plot_single_env(api, entity, project, env, algorithms, valid_seeds, ax=axes[0, idx])
    
    # Plot other environments in second row
    for idx, env in enumerate(other_envs):
        print(f"\n{env}:")
        plot_single_env(api, entity, project, env, algorithms, valid_seeds, ax=axes[1, idx])
    
    plt.suptitle("Performance Comparison Across Environments", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig("Plots/pdf/COMBINED_All_Environments.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("Plots/png/COMBINED_All_Environments.png", dpi=300, bbox_inches="tight")
    print("\nCombined plot saved")
    
    # Collect and print summary statistics
    algo_data = collect_final_scores_and_times(api, entity, project, all_envs, algorithms, valid_seeds)
    print_summary_statistics(algo_data, algorithms)
    
    plt.show()