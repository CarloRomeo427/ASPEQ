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

def get_eval_reward_for_run(api, entity, project, run_name_pattern, metric_name="eval/mean_reward"):
    """Get eval reward data for all seeds of a run."""
    runs = api.runs(f"{entity}/{project}")
    
    all_seeds_data = []
    
    for run in runs:
        if run.name.startswith(run_name_pattern):
            history = run.history(keys=[metric_name, "_step"], pandas=True)
            
            if metric_name in history.columns:
                # Drop NaN values
                history = history.dropna(subset=[metric_name])
                all_seeds_data.append(history[[metric_name, "_step"]])
                print(f"Loaded {run.name}: {len(history)} points")
    
    return all_seeds_data

def compute_mean_std_across_seeds(all_seeds_data, metric_name="eval/mean_reward"):
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

def plot_eval_rewards(entity, project, run_patterns, metric_name="eval/mean_reward", 
                      ema_alpha=0.05, figsize=(10, 6), env_name="Env"):
    """
    Plot eval rewards for multiple runs with EMA smoothing.
    
    Args:
        entity: WandB entity
        project: WandB project name
        run_patterns: List of run name patterns (base names without seed suffix)
        metric_name: Name of eval reward metric
        ema_alpha: EMA smoothing parameter (lower = more smoothing)
        figsize: Figure size tuple
    """
    api = wandb.Api()
    
    plt.figure(figsize=figsize)
    
    for run_pattern in run_patterns:
        print(f"\nProcessing {run_pattern}...")
        
        # Get data for all seeds
        all_seeds_data = get_eval_reward_for_run(api, entity, project, run_pattern, metric_name)
        
        if not all_seeds_data:
            print(f"No data found for {run_pattern}")
            continue
        
        # Compute mean and std
        steps, mean, std = compute_mean_std_across_seeds(all_seeds_data, metric_name)
        
        # Apply EMA
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        run_pattern = run_pattern.split("_")[0].upper()  # Simplify label

        # Plot
        label = run_pattern.replace("_", " ")
        plt.plot(steps, mean_ema, label=label, linewidth=2)
        plt.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.2)
        
        print(f"Final: {mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
    
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Eval Reward", fontsize=12)
    plt.title(f"{env_name.capitalize()}", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.9)
    plt.tight_layout()
    
    
    return plt.gcf()

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    envs = ["Humanoid-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Hopper-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Reacher-v5", "Swimmer-v5", "Pusher-v5"]

    algorithms = ["aspeq", "speq"]  # Add more algorithms as needed

    for env in envs:
        run_patterns = [f"{algo}_{env}" for algo in algorithms]
        # List of run patterns to compare
        
        
        fig = plot_eval_rewards(entity, project, run_patterns, 
                            metric_name="EvalReward",
                            ema_alpha=0.05, env_name=env)
        
        title = ''
        for algo in algorithms:
            title += algo.upper() + 'vs'
        title = title[:-2]  # Remove last ' vs '
        title += f'-{env[:-3]}'
        plt.savefig(f"Plots/pdf/{title}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"Plots/png/{title}.png", dpi=300, bbox_inches="tight")
        print("\nPlot saved to eval_reward_comparison.png")
        plt.show()