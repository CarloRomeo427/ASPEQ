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

def get_eval_reward_for_run(api, entity, project, run_name_pattern, metric_name="EvalReward", valid_seeds=None):
    runs = api.runs(f"{entity}/{project}")
    all_seeds_data = []
    
    for run in runs:
        if run.name == run_name_pattern:
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
                    print(f"  Loaded {run.name} (seed={run.config.get('seed')}): {len(history)} points")
    
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

# Algorithms that always include dataset quality in run name
ALWAYS_SUFFIX_ALGOS = {'calql', 'speq_o2o'}

def build_run_name(algo, env, dataset):
    if algo in ALWAYS_SUFFIX_ALGOS:
        return f"{algo}_{env}_{dataset}"
    else:
        if dataset == "expert":
            return f"{algo}_{env}"
        return f"{algo}_{env}_{dataset}"

def plot_single_env(api, entity, project, env, algorithms, valid_seeds, dataset, 
                    metric_name="EvalReward", ema_alpha=0.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {
        'paspeq_o2o': '#1f77b4',
        'rlpd': '#ff7f0e',
        'calql': '#2ca02c',
        'iql': '#d62728',
        'speq_o2o': '#9467bd',
    }
    
    label_map = {
        'paspeq_o2o': 'PASPEQ O2O (Ours)',
        'rlpd': 'RLPD',
        'calql': 'Cal-QL',
        'iql': 'IQL',
        'speq_o2o': 'SPEQ O2O',
    }
    
    for idx, algo in enumerate(algorithms):
        run_pattern = build_run_name(algo, env, dataset)
        print(f"  Looking for: {run_pattern}")
        
        all_seeds_data = get_eval_reward_for_run(api, entity, project, run_pattern, metric_name, valid_seeds)
        
        if not all_seeds_data:
            print(f"    No data found")
            continue
        
        steps, mean, std = compute_mean_std_across_seeds(all_seeds_data, metric_name)
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        
        algo_label = label_map.get(algo, algo.upper())
        color = color_map.get(algo, plt.cm.tab10(idx))
        
        ax.plot(steps, mean_ema, label=algo_label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.2, color=color)
        print(f"    {algo_label}: {mean_ema[-1]:.2f} ± {std_ema[-1]:.2f}")
    
    env_display = env.replace("-v5", "")
    ax.set_title(f"{env_display}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_ylabel("Eval Reward", fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    return True

def collect_final_scores_and_times(api, entity, project, environments, algorithms, valid_seeds, dataset, metric_name="EvalReward"):
    runs = api.runs(f"{entity}/{project}")
    algo_data = {algo: {'final_scores': [], 'run_times': []} for algo in algorithms}
    
    print("\n" + "="*80)
    print(f"COLLECTING FINAL SCORES AND RUNNING TIMES (dataset={dataset})")
    print("="*80)
    
    for env in environments:
        print(f"\nProcessing {env}...")
        for algo in algorithms:
            run_pattern = build_run_name(algo, env, dataset)
            
            for run in runs:
                if run.name == run_pattern:
                    try:
                        seed = run.config.get('seed', None)
                        if seed not in valid_seeds:
                            continue
                    except:
                        continue
                    
                    history = run.history(keys=[metric_name], pandas=True)
                    if metric_name in history.columns:
                        history = history.dropna(subset=[metric_name])
                        if len(history) > 0:
                            final_score = history[metric_name].iloc[-1]
                            algo_data[algo]['final_scores'].append(final_score)
                            
                            try:
                                runtime_seconds = run.summary.get('_wandb', {}).get('runtime', None)
                                if runtime_seconds is not None:
                                    runtime_hours = runtime_seconds / 3600.0
                                    algo_data[algo]['run_times'].append(runtime_hours)
                            except:
                                pass
    
    return algo_data

def print_summary_statistics(algo_data, algorithms):
    label_map = {
        'paspeq_o2o': 'PASPEQ O2O',
        'rlpd': 'RLPD',
        'calql': 'Cal-QL',
        'iql': 'IQL',
        'speq_o2o': 'SPEQ O2O',
    }
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nFINAL SCORES:")
    for algo in algorithms:
        scores = algo_data[algo]['final_scores']
        label = label_map.get(algo, algo.upper())
        if scores:
            print(f"{label:15s}: {np.mean(scores):8.2f} ± {np.std(scores, ddof=1) if len(scores) > 1 else 0:6.2f}  (n={len(scores)})")
        else:
            print(f"{label:15s}: No data")
    
    print("\nRUNNING TIMES:")
    for algo in algorithms:
        times = algo_data[algo]['run_times']
        label = label_map.get(algo, algo.upper())
        if times:
            print(f"{label:15s}: {np.mean(times):6.2f} ± {np.std(times, ddof=1) if len(times) > 1 else 0:5.2f} hours  (n={len(times)})")
        else:
            print(f"{label:15s}: No timing data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert", choices=["expert", "medium", "simple"])
    parser.add_argument("--combined", action="store_true", help="Plot all envs in single figure")
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    valid_seeds = {0, 42, 1234, 5678, 9876}
    algorithms = ["iql", "speq_o2o", "rlpd", "paspeq_o2o"]
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    if args.dataset == "simple":
        envs = simple_only_envs
    else:
        envs = all_envs
    
    api = wandb.Api()
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {sorted(valid_seeds)}")
    print(f"Environments: {envs}\n")
    
    if args.combined:
        n_envs = len(envs)
        n_cols = min(5, n_envs)
        n_rows = (n_envs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for idx, env in enumerate(envs):
            print(f"\n{env}:")
            plot_single_env(api, entity, project, env, algorithms, valid_seeds, args.dataset, ax=axes[idx])
        
        for idx in range(len(envs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"Performance Comparison - {args.dataset.upper()} Dataset", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/combined_{args.dataset}.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved: combined_{args.dataset}.png")
    else:
        for env in envs:
            print(f"\n{env}:")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_single_env(api, entity, project, env, algorithms, valid_seeds, args.dataset, ax=ax)
            plt.tight_layout()
            plt.savefig(f"Plots/{env}_{args.dataset}.png", dpi=300, bbox_inches="tight")
            plt.close()
        print(f"\nSaved individual plots")
    
    algo_data = collect_final_scores_and_times(api, entity, project, envs, algorithms, valid_seeds, args.dataset)
    print_summary_statistics(algo_data, algorithms)
    
    plt.show()