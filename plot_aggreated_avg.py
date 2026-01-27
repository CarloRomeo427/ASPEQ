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
        if algo == "speq_o2o":
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
                        print(f"  Loaded {run_pattern} seed={seed}: {len(history)} points")
    
    return all_data

def aggregate_across_envs_and_seeds(all_data, n_points=300):
    """Aggregate data across all environments and seeds by interpolating to common steps."""
    if not all_data:
        return None, None, None
    
    max_step = min(d['steps'].max() for d in all_data)
    common_steps = np.linspace(0, max_step, n_points)
    
    interpolated = []
    for d in all_data:
        interp_values = np.interp(common_steps, d['steps'], d['rewards'])
        interpolated.append(interp_values)
    
    stacked = np.array(interpolated)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean)
    
    return common_steps, mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert", choices=["expert", "medium", "simple"])
    parser.add_argument("--ema", type=float, default=0.05, help="EMA smoothing alpha")
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}
    algorithms = ["paspeq_o2o", "rlpd", "speq_o2o"]
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    envs = simple_only_envs if args.dataset == "simple" else all_envs
    
    api = wandb.Api()
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {sorted(valid_seeds)}")
    print(f"Environments: {envs}\n")
    
    color_map = {'paspeq_o2o': '#1f77b4', 'rlpd': '#ff7f0e'}
    label_map = {'paspeq_o2o': 'PASPEQ O2O (Ours)', 'rlpd': 'RLPD'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in algorithms:
        print(f"\nCollecting {algo}...")
        all_data = collect_all_runs(api, entity, project, envs, algo, valid_seeds, args.dataset)
        
        if not all_data:
            print(f"  No data found for {algo}")
            continue
        
        steps, mean, std = aggregate_across_envs_and_seeds(all_data)
        mean_ema = exponential_moving_average(mean, alpha=args.ema)
        std_ema = exponential_moving_average(std, alpha=args.ema)
        
        color = color_map.get(algo, 'gray')
        ax.plot(steps, mean_ema, label=label_map.get(algo, algo.upper()), linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.2, color=color)
        
        print(f"  {algo}: {len(all_data)} runs, final={mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
    
    ax.set_title(f"Average Performance Across All Environments ({args.dataset.upper()})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Eval Reward", fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"Plots/aggregated_{args.dataset}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: aggregated_{args.dataset}.png")
    plt.show()