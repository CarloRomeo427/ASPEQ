import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def exponential_moving_average(data, alpha=0.05):
    """Apply EMA smoothing to data."""
    if len(data) == 0:
        return data
    
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def build_run_name(algo, env, dataset):
    """Build exact run name pattern matching plot_eval_rewards.py logic."""
    if dataset == 'expert':
        return f"{algo}_{env}"
    return f"{algo}_{env}_{dataset}"

def get_run_data_with_gradient_steps(api, entity, project, algo, env, valid_seeds, dataset='expert'):
    """Get eval reward data aligned with cumulative gradient steps for a specific algorithm.
    
    Gradient step costs (CORRECTED from algorithm specifications):
    - RLPD: 1 policy + 20 UTD * 10 Q-nets = 201 updates per env step
    - PASPEQ O2O: 1 policy + 1 UTD * 2 Q-nets = 3 updates per env step + offline epochs * 2
    """
    runs = api.runs(f"{entity}/{project}")
    all_seeds_data = []
    
    # Build exact run name pattern
    run_name_pattern = build_run_name(algo, env, dataset)
    
    for run in runs:
        # Use exact match to avoid matching different datasets
        if run.name != run_name_pattern:
            continue
        
        try:
            seed = run.config.get('seed', None)
            if seed not in valid_seeds:
                continue
        except:
            continue
        
        # Get evaluation rewards and environment steps
        history = run.history(keys=["EvalReward", "_step"], pandas=True)
        history = history.dropna(subset=["EvalReward"])
        
        if len(history) == 0:
            continue
        
        env_steps = history["_step"].values
        rewards = history["EvalReward"].values
        
        # Calculate cumulative gradient steps based on algorithm
        if algo == 'paspeq_o2o':
            # PASPEQ O2O: 1 policy + 1 UTD * 2 Q = 3 per env step + offline epochs * 2
            stab_history = run.history(keys=["stabilization_epochs_performed", "_step"], pandas=True)
            
            # Online: env_steps * 3 (1 policy + 2 Q-nets)
            online_grads = env_steps * 3
            
            # Offline: cumulative offline epochs * 2 Q-nets
            if "stabilization_epochs_performed" in stab_history.columns:
                stab_cumsum = np.zeros_like(env_steps, dtype=float)
                
                for i, step in enumerate(env_steps):
                    mask = stab_history["_step"] <= step
                    stab_cumsum[i] = stab_history.loc[mask, "stabilization_epochs_performed"].sum()
                
                offline_grads = stab_cumsum * 2
            else:
                offline_grads = np.zeros_like(env_steps, dtype=float)
            
            gradient_steps = online_grads + offline_grads
            
        elif algo == 'rlpd':
            # RLPD: 1 policy + 20 UTD * 10 Q-nets = 201 updates per env step
            gradient_steps = env_steps * 201
        
        else:
            gradient_steps = env_steps * 3
        
        df = pd.DataFrame({
            'gradient_steps': gradient_steps,
            'reward': rewards
        })
        all_seeds_data.append(df)
        print(f"      Loaded {run.name} (seed={seed}): {len(df)} points, max gradient steps: {gradient_steps[-1]:.2e}")
    
    return all_seeds_data

def compute_mean_std_across_seeds_gradients(all_seeds_data, max_gradient_steps=None):
    """Compute mean and std across seeds, aligned by gradient steps."""
    if not all_seeds_data:
        return None, None, None
    
    # Collect all gradient steps from all seeds
    all_grad_steps_set = set()
    for df in all_seeds_data:
        all_grad_steps_set.update(df["gradient_steps"].values)
    
    # Sort and convert to array
    all_grad_steps = np.array(sorted(all_grad_steps_set))
    
    # Truncate if needed
    if max_gradient_steps is not None:
        all_grad_steps = all_grad_steps[all_grad_steps <= max_gradient_steps]
    
    if len(all_grad_steps) == 0:
        return None, None, None
    
    # Interpolate each seed to common gradient steps
    interpolated_seeds = []
    for df in all_seeds_data:
        df_sorted = df.sort_values("gradient_steps")
        
        # Only interpolate up to the maximum gradient step this seed reached
        max_seed_grad = df_sorted["gradient_steps"].max()
        valid_grad_steps = all_grad_steps[all_grad_steps <= max_seed_grad]
        
        if len(valid_grad_steps) == 0:
            continue
        
        interp_values = np.interp(valid_grad_steps, 
                                  df_sorted["gradient_steps"].values, 
                                  df_sorted["reward"].values)
        
        # Pad with NaN for gradient steps beyond this seed's range
        full_interp = np.full(len(all_grad_steps), np.nan)
        full_interp[:len(valid_grad_steps)] = interp_values
        
        interpolated_seeds.append(full_interp)
    
    # Stack and compute statistics (ignoring NaN)
    stacked = np.array(interpolated_seeds)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
    
    # Remove trailing NaN values
    valid_mask = ~np.isnan(mean)
    if not np.any(valid_mask):
        return None, None, None
    
    return all_grad_steps[valid_mask], mean[valid_mask], std[valid_mask]

def plot_reward_vs_gradient_steps(entity, project, environments, algorithms, valid_seeds, dataset,
                                   ema_alpha=0.05):
    """Plot reward vs gradient steps for multiple environments with consistent x-axis truncation."""
    api = wandb.Api()
    
    print(f"\nProcessing {len(environments)} environments...")
    print(f"Dataset: {dataset}")
    print(f"Algorithms: {algorithms}")
    
    # First pass: collect all data and find global max gradient steps
    print("\n" + "="*80)
    print("COLLECTING DATA")
    print("="*80)
    
    all_algo_env_data = {}
    
    for env in environments:
        print(f"\n{env}:")
        for algo in algorithms:
            key = (algo, env)
            all_seeds_data = get_run_data_with_gradient_steps(api, entity, project, algo, env, valid_seeds, dataset)
            all_algo_env_data[key] = all_seeds_data
    
    # Don't truncate - let each algorithm show its complete trajectory
    # with proper gradient step scaling. This is how the SPEQ paper does it.
    # RLPD (201x per step) will naturally extend much further than PASPEQ (3x + offline)
    # on the log scale, which correctly shows the computational comparison.
    
    print(f"\n{'='*80}")
    print(f"GRADIENT STEP SCALING (per SPEQ paper methodology):")
    print(f"  PASPEQ O2O: 1 policy + 1 UTD × 2 Q-nets = 3 updates per env step + offline epochs × 2")
    print(f"  RLPD:       1 policy + 20 UTD × 10 Q-nets = 201 updates per env step")
    print(f"Each algorithm shows its full trajectory. RLPD extends ~67x further on x-axis.")
    print(f"{'='*80}")
    
    # Create subplots
    n_envs = len(environments)
    n_cols = min(5, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    print(f"\n{'='*80}")
    print("PLOTTING")
    print(f"{'='*80}")
    
    color_map = {
        'paspeq_o2o': '#1f77b4',
        'rlpd': '#ff7f0e',
    }
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        print(f"\n{env}:")
        
        for algo in algorithms:
            key = (algo, env)
            all_seeds_data = all_algo_env_data.get(key, [])
            
            if not all_seeds_data:
                print(f"  {algo.upper()}: No data")
                continue
            
            # Don't truncate - let each algorithm show full trajectory with proper scaling
            grad_steps, mean, std = compute_mean_std_across_seeds_gradients(
                all_seeds_data, max_gradient_steps=None
            )
            
            if grad_steps is None:
                print(f"  {algo.upper()}: No valid data")
                continue
            
            mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
            std_ema = exponential_moving_average(std, alpha=ema_alpha)
            
            algo_label = "PASPEQ O2O (Ours)" if algo == 'paspeq_o2o' else algo.upper()
            color = color_map.get(algo, '#000000')
            
            ax.plot(grad_steps, mean_ema, label=algo_label, linewidth=2, color=color)
            ax.fill_between(grad_steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.2, color=color)
            print(f"  {algo_label:25s}: gradient range [{grad_steps[0]:.2e}, {grad_steps[-1]:.2e}], final reward = {mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
        
        env_display = env.replace("-v5", "")
        ax.set_title(env_display, fontsize=12, fontweight='bold')
        ax.set_xlabel("Gradient Steps", fontsize=10)
        ax.set_ylabel("Eval Reward", fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    # Hide unused axes
    for idx in range(len(environments), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Reward vs Gradient Steps (Proper Scaling) - {dataset.upper()} Dataset\n" + 
                 f"PASPEQ: 3 steps/env + offline | RLPD: 201 steps/env", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert", choices=["expert", "medium", "simple"])
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    valid_seeds = {0, 42, 1234, 5678, 9876}
    algorithms = ['paspeq_o2o', 'rlpd']
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    # Filter environments based on dataset
    if args.dataset == "simple":
        environments = simple_only_envs
    else:
        environments = all_envs
    
    print(f"Using seeds: {sorted(valid_seeds)}")
    print(f"Algorithms: {algorithms}")
    print(f"Environments: {len(environments)}")
    
    fig = plot_reward_vs_gradient_steps(entity, project, environments, algorithms, valid_seeds, args.dataset)
    
    if fig:
        # Create directories if they don't exist
        os.makedirs("Plots/pdf", exist_ok=True)
        os.makedirs("Plots/png", exist_ok=True)
        
        plt.savefig(f"Plots/pdf/Reward_vs_Gradient_Steps_{args.dataset}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"Plots/png/Reward_vs_Gradient_Steps_{args.dataset}.png", dpi=300, bbox_inches="tight")
        print("\n\nPlot saved")
        plt.show()