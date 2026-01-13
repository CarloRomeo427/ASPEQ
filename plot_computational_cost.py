import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

def build_run_name(algo, env, dataset):
    if dataset == "expert":
        return f"{algo}_{env}"
    return f"{algo}_{env}_{dataset}"

def get_offline_epochs_for_env(api, entity, project, env, dataset, valid_seeds, 
                                metric_name="stabilization_epochs_performed"):
    runs = api.runs(f"{entity}/{project}")
    run_pattern = build_run_name("paspeq_o2o", env, dataset)
    
    all_sums = []
    
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
                total = history[metric_name].sum()
                all_sums.append(total)
                print(f"  {run.name} (seed={seed}): {total:.0f} offline epochs")
    
    if not all_sums:
        return None, None
    
    return np.mean(all_sums), np.std(all_sums, ddof=1) if len(all_sums) > 1 else 0

def plot_computational_comparison(entity, project, environments, dataset, valid_seeds):
    api = wandb.Api()
    
    # =========================================================================
    # RLPD PARAMETERS
    # =========================================================================
    ONLINE_STEPS = 300_000
    RLPD_UTD = 20
    RLPD_NUM_Q = 10
    RLPD_POLICY_UPDATE_DELAY = 20
    
    # RLPD gradient steps:
    # - Q updates: ONLINE_STEPS * UTD * NUM_Q (each of UTD updates, updates all NUM_Q networks)
    # - Policy updates: ONLINE_STEPS * UTD / POLICY_UPDATE_DELAY
    rlpd_online_q = ONLINE_STEPS * RLPD_UTD * RLPD_NUM_Q  # 60,000,000
    rlpd_online_policy = ONLINE_STEPS * RLPD_UTD // RLPD_POLICY_UPDATE_DELAY  # 300,000
    rlpd_total = rlpd_online_q + rlpd_online_policy  # 60,300,000
    
    # =========================================================================
    # PASPEQ O2O PARAMETERS
    # =========================================================================
    PASPEQ_UTD = 1
    PASPEQ_NUM_Q = 2
    PASPEQ_POLICY_UPDATE_DELAY = 20
    
    # PASPEQ online gradient steps:
    # - Q updates: ONLINE_STEPS * UTD * NUM_Q = 300,000 * 1 * 2 = 600,000
    # - Policy updates: With UTD=1, policy_update_delay=20, but since i_update == num_update-1
    #   is always true when UTD=1, policy updates every env step ≈ 300,000
    paspeq_online_q = ONLINE_STEPS * PASPEQ_UTD * PASPEQ_NUM_Q  # 600,000
    paspeq_online_policy = ONLINE_STEPS  # ~300,000 (policy updates every step with UTD=1)
    paspeq_online_total = paspeq_online_q + paspeq_online_policy  # 900,000
    
    print("="*80)
    print(f"COMPUTATIONAL COST ANALYSIS (dataset={dataset})")
    print("="*80)
    print(f"\nRLPD breakdown:")
    print(f"  Online Q updates:      {rlpd_online_q:>12,} (300k × 20 UTD × 10 Q-nets)")
    print(f"  Online Policy updates: {rlpd_online_policy:>12,} (300k × 20 / 20 delay)")
    print(f"  TOTAL:                 {rlpd_total:>12,}")
    print(f"\nPASPEQ O2O online breakdown:")
    print(f"  Online Q updates:      {paspeq_online_q:>12,} (300k × 1 UTD × 2 Q-nets)")
    print(f"  Online Policy updates: {paspeq_online_policy:>12,} (300k, every step with UTD=1)")
    print(f"  Online TOTAL:          {paspeq_online_total:>12,}")
    print(f"  + Offline Q updates:   offline_epochs × 2")
    
    env_data = {}
    
    for env in environments:
        print(f"\n{env}:")
        mean_epochs, std_epochs = get_offline_epochs_for_env(
            api, entity, project, env, dataset, valid_seeds
        )
        
        if mean_epochs is None:
            print(f"  No data found")
            continue
        
        # CORRECTED FORMULA:
        # Total = online_Q + online_policy + offline_Q
        #       = 600,000 + 300,000 + (offline_epochs × 2)
        #       = 900,000 + (offline_epochs × 2)
        offline_q_steps = mean_epochs * PASPEQ_NUM_Q
        offline_q_std = std_epochs * PASPEQ_NUM_Q
        
        paspeq_total = paspeq_online_total + offline_q_steps
        paspeq_std = offline_q_std  # Online is constant, only offline varies
        
        env_data[env] = {
            'mean': paspeq_total,
            'std': paspeq_std,
            'offline_epochs_mean': mean_epochs,
            'offline_epochs_std': std_epochs,
            'offline_q_steps': offline_q_steps
        }
        
        print(f"  Offline epochs:        {mean_epochs:>12,.0f} ± {std_epochs:,.0f}")
        print(f"  Offline Q updates:     {offline_q_steps:>12,.0f} (epochs × 2)")
        print(f"  PASPEQ O2O TOTAL:      {paspeq_total:>12,.0f} ± {paspeq_std:,.0f}")
    
    if not env_data:
        print("No valid data found!")
        return None, None, None
    
    env_names = [e.replace("-v5", "") for e in env_data.keys()]
    means = [env_data[e]['mean'] for e in env_data.keys()]
    stds = [env_data[e]['std'] for e in env_data.keys()]
    x_pos = np.arange(len(env_names))
    
    # Figure 1: With RLPD line
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x_pos, means, yerr=stds, capsize=5, color='#1f77b4', 
            alpha=0.8, edgecolor='black', linewidth=1.5, label='PASPEQ O2O')
    ax1.axhline(y=rlpd_total, color='#ff7f0e', linestyle='--', linewidth=2.5, label='RLPD')
    ax1.set_ylabel("Total Gradient Steps", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Environment", fontsize=12, fontweight='bold')
    ax1.set_title(f"Computational Cost: PASPEQ O2O vs RLPD ({dataset.upper()} dataset)", 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(env_names, fontsize=10, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax1.set_ylim(bottom=0)
    plt.tight_layout()
    
    # Figure 2: Without RLPD line (zoomed to PASPEQ scale)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(x_pos, means, yerr=stds, capsize=5, color='#1f77b4', 
            alpha=0.8, edgecolor='black', linewidth=1.5, label='PASPEQ O2O')
    ax2.set_ylabel("Total Gradient Steps", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Environment", fontsize=12, fontweight='bold')
    ax2.set_title(f"PASPEQ O2O Computational Cost ({dataset.upper()} dataset)", 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(env_names, fontsize=10, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    
    # Figure 3: Summary table
    fig3, ax3 = plt.subplots(figsize=(12, max(3, len(env_data) * 0.5 + 2)))
    ax3.axis('off')
    
    table_data = []
    for env in env_data.keys():
        env_name = env.replace("-v5", "")
        paspeq_steps = env_data[env]['mean']
        paspeq_std = env_data[env]['std']
        offline_epochs = env_data[env]['offline_epochs_mean']
        reduction = (1 - paspeq_steps / rlpd_total) * 100
        table_data.append([
            env_name,
            f"{offline_epochs:,.0f}",
            f"{paspeq_steps:,.0f} ± {paspeq_std:,.0f}",
            f"{reduction:.1f}%"
        ])
    
    # Add average row
    avg_paspeq = np.mean(means)
    avg_offline = np.mean([env_data[e]['offline_epochs_mean'] for e in env_data.keys()])
    avg_reduction = (1 - avg_paspeq / rlpd_total) * 100
    table_data.append(["Average", f"{avg_offline:,.0f}", f"{avg_paspeq:,.0f}", f"{avg_reduction:.1f}%"])
    
    table = ax3.table(
        cellText=table_data,
        colLabels=["Environment", "Offline Epochs", "PASPEQ O2O Total Steps", "Reduction vs RLPD"],
        loc='center',
        cellLoc='center',
        colWidths=[0.22, 0.22, 0.34, 0.22]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style average row
    for j in range(4):
        table[(len(table_data), j)].set_facecolor('#D6DCE5')
        table[(len(table_data), j)].set_text_props(fontweight='bold')
    
    title_text = (f"PASPEQ O2O vs RLPD Summary ({dataset.upper()} dataset)\n"
                  f"RLPD: {rlpd_total:,} steps | "
                  f"PASPEQ Online: {paspeq_online_total:,} steps + Offline Q updates")
    ax3.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (CORRECTED)")
    print("="*80)
    print(f"\nRLPD (constant):         {rlpd_total:>12,} gradient steps")
    print(f"PASPEQ online (constant): {paspeq_online_total:>12,} gradient steps")
    print(f"\nPer-environment breakdown:")
    print(f"{'Environment':<25} {'Offline Epochs':>15} {'Total Steps':>20} {'Reduction':>12}")
    print("-"*75)
    for env, data in env_data.items():
        reduction = (1 - data['mean'] / rlpd_total) * 100
        print(f"{env:<25} {data['offline_epochs_mean']:>15,.0f} {data['mean']:>20,.0f} {reduction:>11.1f}%")
    print("-"*75)
    print(f"{'Average':<25} {avg_offline:>15,.0f} {avg_paspeq:>20,.0f} {avg_reduction:>11.1f}%")
    
    return fig1, fig2, fig3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert", choices=["expert", "medium", "simple"])
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    valid_seeds = {0, 42, 1234, 5678, 9876, 777, 24680, 13579, 31415, 27182}
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    envs = simple_only_envs if args.dataset == "simple" else all_envs
    
    fig1, fig2, fig3 = plot_computational_comparison(entity, project, envs, args.dataset, valid_seeds)
    
    if fig1 is not None:
        fig1.savefig(f"computational_cost_{args.dataset}_with_rlpd.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"computational_cost_{args.dataset}_paspeq_only.png", dpi=300, bbox_inches="tight")
        fig3.savefig(f"computational_cost_{args.dataset}_table.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved: computational_cost_{args.dataset}_with_rlpd.png")
        print(f"Saved: computational_cost_{args.dataset}_paspeq_only.png")
        print(f"Saved: computational_cost_{args.dataset}_table.png")
        plt.show()