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

def build_run_name(algo, env, dataset):
    if dataset == "expert":
        if algo != "rlpd":
            return f"{algo}_{env}_{dataset}"
        else:
            return f"{algo}_{env}"
    return f"{algo}_{env}_{dataset}"

def collect_all_runs(api, entity, project, environments, algo, valid_seeds, dataset, metric_name="EvalReward"):
    """Collect all runs for an algorithm across all environments and seeds."""
    runs = api.runs(f"{entity}/{project}")
    all_data = []
    
    for env in environments:
        run_pattern = build_run_name(algo, env, dataset)
        
        for run in runs:
            if run.name == run_pattern:
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
                        all_data.append({
                            'env': env,
                            'seed': seed,
                            'steps': history["_step"].values,
                            'rewards': history[metric_name].values
                        })
                        print(f"    Loaded {run_pattern} seed={seed}: {len(history)} points, max_step={history['_step'].max()}")
    
    return all_data

def aggregate_across_envs_and_seeds_full(all_data, n_points=300):
    """
    Aggregate data across all environments and seeds.
    Uses the MAXIMUM step range and handles missing data by using available runs at each point.
    """
    if not all_data:
        return None, None, None
    
    # Use MAXIMUM step across all runs (not minimum)
    max_step = max(d['steps'].max() for d in all_data)
    common_steps = np.linspace(0, max_step, n_points)
    
    # For each step, collect values from runs that have data up to that point
    means_at_step = []
    stds_at_step = []
    
    for step in common_steps:
        values_at_step = []
        for d in all_data:
            # Only include this run if it has data covering this step
            if d['steps'].max() >= step:
                # Interpolate to get value at this step
                interp_val = np.interp(step, d['steps'], d['rewards'])
                values_at_step.append(interp_val)
        
        if len(values_at_step) > 0:
            means_at_step.append(np.mean(values_at_step))
            stds_at_step.append(np.std(values_at_step, ddof=1) if len(values_at_step) > 1 else 0.0)
        else:
            means_at_step.append(np.nan)
            stds_at_step.append(np.nan)
    
    return common_steps, np.array(means_at_step), np.array(stds_at_step)

def aggregate_across_datasets_full(dataset_results, n_points=300):
    """
    Aggregate results across multiple datasets.
    Uses MAXIMUM step range and handles missing data gracefully.
    """
    valid_results = {k: v for k, v in dataset_results.items() if v[0] is not None}
    
    if not valid_results:
        return None, None, None
    
    # Use MAXIMUM step across all datasets
    max_step = max(v[0][~np.isnan(v[1])].max() if np.any(~np.isnan(v[1])) else 0 
                   for v in valid_results.values())
    common_steps = np.linspace(0, max_step, n_points)
    
    # For each step, average across datasets that have valid data
    means_at_step = []
    stds_at_step = []
    
    for step in common_steps:
        values_at_step = []
        for dataset_name, (steps, mean, std) in valid_results.items():
            # Find valid (non-nan) range for this dataset
            valid_mask = ~np.isnan(mean)
            if not np.any(valid_mask):
                continue
            valid_steps = steps[valid_mask]
            valid_mean = mean[valid_mask]
            
            # Only include if this dataset covers this step
            if valid_steps.max() >= step:
                interp_val = np.interp(step, valid_steps, valid_mean)
                values_at_step.append(interp_val)
        
        if len(values_at_step) > 0:
            means_at_step.append(np.mean(values_at_step))
            stds_at_step.append(np.std(values_at_step, ddof=1) if len(values_at_step) > 1 else 0.0)
        else:
            means_at_step.append(np.nan)
            stds_at_step.append(np.nan)
    
    return common_steps, np.array(means_at_step), np.array(stds_at_step)

def auto_label(algo):
    """Auto-generate a clean label from algorithm name."""
    # Direct mappings for common names
    direct = {
        'rlpd': 'RLPD',
        'iql': 'IQL',
        'calql': 'Cal-QL',
        'sacfd': 'SACfD',
        'sac': 'SAC',
    }
    if algo in direct:
        return direct[algo]
    
    # Handle patterns
    label = algo
    label = label.replace('_o2o', ' O2O')
    label = label.replace('_pi', ' (π)')
    label = label.replace('_td', ' (TD)')
    label = label.replace('pct10', '10%')
    label = label.replace('pct20', '20%')
    label = label.replace('pct', '%')
    label = label.replace('valpat', 'pat=')
    label = label.replace('_', ' ')
    
    # Clean up
    while '  ' in label:
        label = label.replace('  ', ' ')
    
    # Capitalize properly
    parts = label.split()
    result = []
    for p in parts:
        if p.upper() in ['O2O', 'TD', 'RLPD', 'IQL', 'SAC']:
            result.append(p.upper())
        elif p.startswith('(') or p.endswith(')') or p.endswith('%') or p.startswith('pat='):
            result.append(p)
        else:
            result.append(p.upper())
    
    return ' '.join(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ema", type=float, default=0.05, help="EMA smoothing alpha (lower=smoother)")
    parser.add_argument("--std-alpha", type=float, default=0.1, help="Std shading opacity")
    parser.add_argument("--per-dataset", action="store_true", help="Also plot per-dataset figures")
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}
    
    # ============================================================
    # ONLY CHANGE THIS LIST - everything else is automatic
    # ============================================================
    algorithms = [
        "speq_o2o", 
        "rlpd", 
        "faspeq_pct10_pi", 
        "faspeq_pct20_pi", 
        "sacfd",
        "paspeq_o2o",
        "faspeq_o2o"
    ]
    # ============================================================
    
    datasets = ["expert", "medium", "simple"]
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    api = wandb.Api()
    
    # Auto-generate colors and labels
    cmap = plt.cm.tab10
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    label_map = {algo: auto_label(algo) for algo in algorithms}
    
    # Print what we're plotting
    print("="*80)
    print("ALGORITHMS TO PLOT:")
    for algo in algorithms:
        print(f"  {algo} -> {label_map[algo]}")
    print("="*80)
    
    print("\nCOLLECTING DATA ACROSS ALL DATASETS (FULL CURVES)")
    print("="*80)
    
    # Store per-dataset results for each algorithm
    algo_dataset_results = {algo: {} for algo in algorithms}
    
    for dataset in datasets:
        envs = simple_only_envs if dataset == "simple" else all_envs
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset.upper()}")
        print(f"Environments: {len(envs)}")
        print(f"{'='*40}")
        
        for algo in algorithms:
            print(f"\n  Collecting {algo}...")
            all_data = collect_all_runs(api, entity, project, envs, algo, valid_seeds, dataset)
            
            if not all_data:
                print(f"    No data found for {algo}")
                algo_dataset_results[algo][dataset] = (None, None, None)
                continue
            
            # Report max steps per run
            max_steps = [d['steps'].max() for d in all_data]
            print(f"    {algo}: {len(all_data)} runs")
            print(f"    Step ranges: min={min(max_steps):.0f}, max={max(max_steps):.0f}")
            
            steps, mean, std = aggregate_across_envs_and_seeds_full(all_data)
            algo_dataset_results[algo][dataset] = (steps, mean, std)
            
            # Report final value (at last non-nan point)
            valid_mask = ~np.isnan(mean)
            if np.any(valid_mask):
                last_valid_idx = np.where(valid_mask)[0][-1]
                print(f"    Final (step {steps[last_valid_idx]:.0f}): {mean[last_valid_idx]:.2f} ± {std[last_valid_idx]:.2f}")
    
    # Aggregate across all datasets for each algorithm
    print("\n" + "="*80)
    print("AGGREGATING ACROSS ALL DATASETS (FULL CURVES)")
    print("="*80)
    
    algo_final_results = {}
    
    for algo in algorithms:
        print(f"\n{algo}:")
        dataset_results = algo_dataset_results[algo]
        
        valid_count = sum(1 for v in dataset_results.values() if v[0] is not None)
        print(f"  Valid datasets: {valid_count}/{len(datasets)}")
        
        if valid_count == 0:
            print(f"  Skipping {algo} - no valid data")
            continue
        
        steps, mean, std = aggregate_across_datasets_full(dataset_results)
        algo_final_results[algo] = (steps, mean, std)
        
        valid_mask = ~np.isnan(mean)
        if np.any(valid_mask):
            last_valid_idx = np.where(valid_mask)[0][-1]
            print(f"  Final (step {steps[last_valid_idx]:.0f}): {mean[last_valid_idx]:.2f} ± {std[last_valid_idx]:.2f}")
    
    # Plot aggregated across all datasets
    print("\n" + "="*80)
    print("PLOTTING AGGREGATE OF AGGREGATES (FULL CURVES)")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        if algo not in algo_final_results:
            continue
        
        steps, mean, std = algo_final_results[algo]
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(mean)
        steps_valid = steps[valid_mask]
        mean_valid = mean[valid_mask]
        std_valid = std[valid_mask]
        
        mean_ema = exponential_moving_average(mean_valid, alpha=args.ema)
        std_ema = exponential_moving_average(std_valid, alpha=args.ema)
        
        color = color_map[algo]
        label = label_map[algo]
        
        ax.plot(steps_valid, mean_ema, label=label, linewidth=2, color=color)
        ax.fill_between(steps_valid, mean_ema - std_ema, mean_ema + std_ema, alpha=args.std_alpha, color=color)
        
        print(f"  {label}: plotted {len(steps_valid)} points, max_step={steps_valid.max():.0f}, final={mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
    
    ax.set_title("Average Performance Across All Environments and Datasets\n(Expert + Medium + Simple)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Eval Reward", fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Plots/aggregated_all_datasets.png", dpi=300, bbox_inches="tight")
    plt.savefig("Plots/pdf/aggregated_all_datasets.pdf", dpi=300, bbox_inches="tight")
    print(f"\nSaved: Plots/aggregated_all_datasets.png")
    print(f"Saved: Plots/pdf/aggregated_all_datasets.pdf")
    
    # Optionally plot per-dataset figures
    if args.per_dataset:
        print("\n" + "="*80)
        print("PLOTTING PER-DATASET FIGURES (FULL CURVES)")
        print("="*80)
        
        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for algo in algorithms:
                result = algo_dataset_results[algo].get(dataset)
                if result is None or result[0] is None:
                    continue
                
                steps, mean, std = result
                
                # Remove NaN values
                valid_mask = ~np.isnan(mean)
                steps_valid = steps[valid_mask]
                mean_valid = mean[valid_mask]
                std_valid = std[valid_mask]
                
                if len(steps_valid) == 0:
                    continue
                
                mean_ema = exponential_moving_average(mean_valid, alpha=args.ema)
                std_ema = exponential_moving_average(std_valid, alpha=args.ema)
                
                color = color_map[algo]
                label = label_map[algo]
                
                ax.plot(steps_valid, mean_ema, label=label, linewidth=2, color=color)
                ax.fill_between(steps_valid, mean_ema - std_ema, mean_ema + std_ema, alpha=args.std_alpha, color=color)
            
            ax.set_title(f"Average Performance Across All Environments ({dataset.upper()})", 
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Eval Reward", fontsize=12)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"Plots/aggregated_{dataset}.png", dpi=300, bbox_inches="tight")
            print(f"Saved: Plots/aggregated_{dataset}.png")
            plt.close()
    
    plt.show()