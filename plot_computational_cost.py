import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_valid_seeds_for_env(api, entity, project, env, algorithms, metric_name="EvalReward", expected_points=300):
    """
    Get seeds that are completed for ALL algorithms in the list.
    """
    runs = api.runs(f"{entity}/{project}")
    
    algo_seeds = {algo: set() for algo in algorithms}
    
    for run in runs:
        parts = run.name.split("_")
        if len(parts) < 2:
            continue
        
        algo = parts[0]
        if algo not in algorithms:
            continue
        
        env_part = "_".join(parts[1:])
        if not env_part.startswith(env):
            continue
        
        try:
            seed = run.config.get('seed', None)
            if seed is None:
                continue
        except:
            continue
        
        history = run.history(keys=[metric_name], pandas=True)
        if metric_name in history.columns:
            eval_points = history[metric_name].dropna().shape[0]
            if eval_points >= expected_points:
                algo_seeds[algo].add(seed)
    
    if not algo_seeds:
        return set()
    
    valid_seeds = set.intersection(*algo_seeds.values())
    return valid_seeds

def get_global_average_stabilization(api, entity, project, environments, algorithms=["aspeq", "speq"],
                                    metric_name="stabilization_epochs_performed"):
    """
    Calculate global average stabilization epochs across all environments and seeds for ASPEQ.
    
    Returns:
        tuple: (mean, std, all individual sums across all envs/seeds)
    """
    all_sums = []
    
    for env in environments:
        print(f"\nProcessing {env}...")
        
        # Get valid seeds for this environment
        valid_seeds = get_valid_seeds_for_env(api, entity, project, env, algorithms)
        
        if not valid_seeds:
            print(f"  No valid seeds for {env}, skipping")
            continue
        
        print(f"  Valid seeds: {sorted(valid_seeds)}")
        
        # Get all ASPEQ runs for this environment
        runs = api.runs(f"{entity}/{project}")
        
        for run in runs:
            if run.name.startswith(f"aspeq_{env}"):
                # Check if seed is valid
                try:
                    seed = run.config.get('seed', None)
                    if seed not in valid_seeds:
                        continue
                except:
                    continue
                
                # Sum stabilization epochs for this run
                history = run.history(keys=[metric_name], pandas=True)
                
                if metric_name in history.columns:
                    total = history[metric_name].sum()
                    all_sums.append(total)
                    print(f"  {run.name}: {total:.0f}")
    
    if not all_sums:
        print("\nNo valid data found!")
        return None, None, []
    
    mean = np.mean(all_sums)
    std = np.std(all_sums, ddof=1) if len(all_sums) > 1 else 0
    
    return mean, std, all_sums

def plot_global_comparison(entity, project, environments):
    """
    Plot global comparison of total computational cost across algorithms.
    """
    api = wandb.Api()
    
    # Calculate ASPEQ average stabilization
    print("="*80)
    print("CALCULATING ASPEQ GLOBAL AVERAGE")
    print("="*80)
    
    aspeq_stab_mean, aspeq_stab_std, all_sums = get_global_average_stabilization(
        api, entity, project, environments
    )
    
    if aspeq_stab_mean is None:
        print("ERROR: No valid ASPEQ data found!")
        return None
    
    print(f"\n{'='*80}")
    print(f"ASPEQ Stabilization Statistics:")
    print(f"  Total runs analyzed: {len(all_sums)}")
    print(f"  Mean stabilization epochs: {aspeq_stab_mean:.2e} ± {aspeq_stab_std:.2e}")
    print(f"{'='*80}\n")
    
    # Constants
    ONLINE_INTERACTIONS = 300000
    SPEQ_STABILIZATION = 30 * 75000  # 2,250,000
    
    # Total costs
    ### algo_cost = policy updates + q updates * num_q
    sac_total = ONLINE_INTERACTIONS + ONLINE_INTERACTIONS * 2
    aspeq_total = ONLINE_INTERACTIONS + ONLINE_INTERACTIONS * 2 + aspeq_stab_mean * 2
    aspeq_total_std = aspeq_stab_std * 2  # std scales with multiplier
    speq_total = ONLINE_INTERACTIONS + ONLINE_INTERACTIONS * 2 + SPEQ_STABILIZATION * 2
    droq_total = ONLINE_INTERACTIONS + ONLINE_INTERACTIONS * 20 * 2
    # redq_total = ONLINE_INTERACTIONS + ONLINE_INTERACTIONS * 20 * 20
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    sac_color = '#2ca02c'      # Green
    aspeq_color = '#1f77b4'    # Blue
    speq_color = '#ff7f0e'     # Orange
    droq_color = '#d62728'     # Red
    # redq_color = '#9467bd'     # Purple
    
    # Data
    algorithms = ['SAC', 'ASPEQ', 'SPEQ', 'DroQ']
    values = [sac_total, aspeq_total, speq_total, droq_total ]
    colors = [sac_color, aspeq_color, speq_color, droq_color ]
    x_pos = [0, 1, 2, 3]
    
    # Plot bars with edges and rounded corners (using edgecolor and linewidth)
    bars = ax.bar(x_pos, values, color=colors, alpha=0.85, width=0.5,
                  edgecolor='black', linewidth=2.5)
    
    # Manually round the corners by adjusting the bar patches
    from matplotlib.patches import FancyBboxPatch
    
    for bar in bars:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Remove the original bar
        bar.remove()
        
        # Create a rounded rectangle patch
        rounded_bar = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black',
            linewidth=2.5, alpha=0.85
        )
        ax.add_patch(rounded_bar)
    
    # Format
    ax.set_ylabel("Total Updates (Gradient Steps)", fontsize=14, fontweight='bold')
    ax.set_title("Total Computational Cost Comparison", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL COMPARISON:")
    print("="*80)
    print(f"SAC:   {sac_total:,.0f}")
    print(f"ASPEQ: {aspeq_total:,.0f} ± {aspeq_total_std:,.0f}")
    print(f"SPEQ:  {speq_total:,.0f}")
    print(f"DroQ:  {droq_total:,.0f}")
    # print(f"RedQ:  {redq_total:,.0f}")
    print(f"\nASPEQ vs SPEQ reduction: {(1 - aspeq_total/speq_total)*100:.1f}%")
    print(f"ASPEQ overhead vs SAC: {(aspeq_total/sac_total - 1)*100:.1f}%")
    print(f"ASPEQ vs DroQ reduction: {(1 - aspeq_total/droq_total)*100:.1f}%")
    # print(f"ASPEQ vs RedQ reduction: {(1 - aspeq_total/redq_total)*100:.1f}%")
    print("="*80)
    
    return fig

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    environments = [
        "Humanoid-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Hopper-v5",
        "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Reacher-v5", 
        "Swimmer-v5", "Pusher-v5"
    ]
    
    fig = plot_global_comparison(entity, project, environments)
    
    if fig is not None:
        # Save
        plt.savefig("Plots/pdf/Global_Computational_Cost_Comparison.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("Plots/png/Global_Computational_Cost_Comparison.png", dpi=300, bbox_inches="tight")
        print("\nPlot saved")
        plt.show()