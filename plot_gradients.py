import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha=0.05):
    """Apply EMA smoothing to data."""
    if len(data) == 0:
        return data
    
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def get_run_data_with_gradient_steps(api, entity, project, algo, env, valid_seeds):
    """Get eval reward data aligned with cumulative gradient steps for a specific algorithm."""
    runs = api.runs(f"{entity}/{project}")
    all_seeds_data = []
    
    for run in runs:
        if run.name.startswith(f"{algo}_{env}"):
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
            if algo == 'sac':
                # SAC: 1 policy update + 2 Q updates per env step
                gradient_steps = env_steps + env_steps * 2
                
            elif algo == 'droq':
                # DroQ: 1 policy update + 20 UTD * 2 Q networks
                gradient_steps = env_steps + env_steps * 20 * 2
                
            elif algo == 'redq':
                # RedQ: 1 policy update + 20 UTD * 20 Q networks
                gradient_steps = env_steps + env_steps * 20 * 20
                
            elif algo == 'speq':
                # SPEQ: 1 policy + 2 Q online + periodic offline phases
                online_grads = env_steps + env_steps * 2
                # Offline: every 10k steps, do 75k epochs on 2 Q networks
                num_offline_phases = np.floor(env_steps / 10000)
                offline_grads = num_offline_phases * 75000 * 2
                gradient_steps = online_grads + offline_grads
                
            elif algo == 'aspeq':
                # ASPEQ: 1 policy + 2 Q online + dynamic offline phases
                stab_history = run.history(keys=["stabilization_epochs_performed", "_step"], pandas=True)
                
                online_grads = env_steps + env_steps * 2
                
                if "stabilization_epochs_performed" in stab_history.columns:
                    stab_cumsum = np.zeros_like(env_steps, dtype=float)
                    
                    for i, step in enumerate(env_steps):
                        mask = stab_history["_step"] <= step
                        stab_cumsum[i] = stab_history.loc[mask, "stabilization_epochs_performed"].sum()
                    
                    offline_grads = stab_cumsum * 2
                else:
                    offline_grads = 0
                
                gradient_steps = online_grads + offline_grads
            
            else:
                gradient_steps = env_steps * 3
            
            df = pd.DataFrame({
                'gradient_steps': gradient_steps,
                'reward': rewards
            })
            all_seeds_data.append(df)
            print(f"  Loaded {run.name}: {len(df)} points, max gradient steps: {gradient_steps[-1]:.2e}")
    
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
    std = np.nanstd(stacked, axis=0, ddof=1)
    
    # Remove trailing NaN values
    valid_mask = ~np.isnan(mean)
    if not np.any(valid_mask):
        return None, None, None
    
    return all_grad_steps[valid_mask], mean[valid_mask], std[valid_mask]

def plot_reward_vs_gradient_steps(entity, project, environments, algorithms, valid_seeds, 
                                   ema_alpha=0.05, truncate_at='aspeq'):
    """Plot reward vs gradient steps for multiple environments."""
    api = wandb.Api()
    
    # Color mapping - matching W&B colors more closely
    color_map = {
        'sac': '#2ca02c',        # Green (like W&B)
        'droq': '#d62728',       # Red (like W&B)
        'aspeq': '#9467bd',      # Purple/Blue (like W&B)
        'speq': '#ff7f0e',       # Orange (like W&B)
        'redq': '#8c564b'        # Brown
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        print(f"\n{'='*80}")
        print(f"Processing {env}")
        print(f"{'='*80}")
        
        # Collect all data first
        algo_data = {}
        max_end_aspeq = 0
        max_end_speq = 0
        
        for algo in algorithms:
            print(f"\n{algo.upper()}:")
            all_seeds_data = get_run_data_with_gradient_steps(api, entity, project, algo, env, valid_seeds)
            
            if not all_seeds_data:
                print(f"  No data found for {algo}_{env}")
                continue
            
            algo_data[algo] = all_seeds_data
            
            # Track max gradient steps for truncation
            for df in all_seeds_data:
                max_grad = df['gradient_steps'].max()
                if algo == 'aspeq':
                    max_end_aspeq = max(max_end_aspeq, max_grad)
                elif algo == 'speq':
                    max_end_speq = max(max_end_speq, max_grad)
        
        if not algo_data:
            print(f"No data for any algorithm in {env}")
            continue
        
        # Determine truncation point
        max_gradient_steps = max_end_aspeq if truncate_at == 'aspeq' else max_end_speq
        print(f"\nTruncating at {truncate_at.upper()}: {max_gradient_steps:.2e} gradient steps")
        
        # Plot each algorithm
        for algo in algorithms:
            if algo not in algo_data:
                continue
            
            # Compute mean and std
            grad_steps, mean, std = compute_mean_std_across_seeds_gradients(
                algo_data[algo], max_gradient_steps=max_gradient_steps
            )
            
            if grad_steps is None:
                print(f"{algo.upper()}: No valid data after truncation")
                continue
            
            # Apply EMA smoothing
            mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
            std_ema = exponential_moving_average(std, alpha=ema_alpha)
            
            # Plot
            color = color_map.get(algo, '#000000')
            label = 'SPEQ (Ours)' if algo == 'speq' else algo.upper()
            
            ax.plot(grad_steps, mean_ema, label=label, linewidth=2, color=color)
            ax.fill_between(grad_steps, mean_ema - std_ema, mean_ema + std_ema, 
                           alpha=0.2, color=color)
            
            print(f"{algo.upper()}: Final = {mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
        
        # Format subplot
        env_display = env.replace("-v5", "")
        ax.set_title(env_display, fontsize=14, fontweight='bold')
        ax.set_xlabel("Gradient Steps", fontsize=11)
        ax.set_ylabel("Eval Reward", fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    truncate_label = "ASPEQ" if truncate_at == 'aspeq' else "SPEQ"
    plt.suptitle(f"Reward vs Gradient Steps Comparison (Truncated at {truncate_label})", 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    valid_seeds = {0, 42, 1234, 5678, 777, 9876, 13579, 31415, 24680, 27182}
    
    environments = [
        "Ant-v5", "Walker2d-v5", "HalfCheetah-v5", "Humanoid-v5", "Hopper-v5",
        "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Reacher-v5", 
        "Swimmer-v5", "Pusher-v5"
    ]
    
    algorithms = ['sac', 'paspeq', 'speq', 'droq']
    
    TRUNCATE_AT = 'aspeq'
    
    print(f"Using seeds: {sorted(valid_seeds)}")
    print(f"Truncating at: {TRUNCATE_AT.upper()}\n")
    
    fig = plot_reward_vs_gradient_steps(entity, project, environments, algorithms, valid_seeds,
                                         truncate_at=TRUNCATE_AT)
    
    filename_suffix = f"_truncated_at_{TRUNCATE_AT.upper()}"
    plt.savefig(f"Plots/pdf/Reward_vs_Gradient_Steps{filename_suffix}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"Plots/png/Reward_vs_Gradient_Steps{filename_suffix}.png", dpi=300, bbox_inches="tight")
    print("\n\nPlot saved")
    plt.show()