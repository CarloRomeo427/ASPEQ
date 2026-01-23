"""
Computational Cost Comparison Table Generator

Computes total gradient steps for each algorithm across MuJoCo environments.

Gradient step formulas:
┌─────────────┬──────────────────────────────────────────────────────────────────────────────┐
│ Algorithm   │ Total Gradient Steps Formula                                                 │
├─────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ RLPD        │ online_steps × 201  (1 policy + 20 UTD × 10 Q-nets)                          │
│ PASPEQ O2O  │ online_steps × 3 + offline_epochs × 2  (dynamic offline from WandB)         │
│ SPEQ        │ online_steps × 3 + (online_steps/10000) × 75000 × 2  (fixed 75k every 10k)  │
│ IQL         │ online_steps × 4 + offline_pretrain × 4  (1V + 2Q + 1π per step)            │
│ Cal-QL      │ online_steps × 3 + offline_pretrain × 3  (2Q + 1π per step)                 │
│ SAC         │ online_steps × 3  (1 policy + 2 Q-nets, UTD=1)                              │
└─────────────┴──────────────────────────────────────────────────────────────────────────────┘

Configuration:
- Online training steps: 300,000 for all algorithms
- IQL/CalQL offline pretraining: 300,000 steps (configurable)
- RLPD: UTD=20, 10 Q-networks
- PASPEQ/SPEQ/CalQL/SAC: UTD=1, 2 Q-networks
- IQL: UTD=1, 2 Q-networks + 1 Value network
"""

import wandb
import pandas as pd
import numpy as np
import argparse

def get_paspeq_offline_epochs(api, entity, project, env, valid_seeds, dataset='expert'):
    """Fetch total offline epochs performed by PASPEQ O2O from WandB logs."""
    runs = api.runs(f"{entity}/{project}")
    
    if dataset == 'expert':
        run_name_pattern = f"paspeq_o2o_{env}"
    else:
        run_name_pattern = f"paspeq_o2o_{env}_{dataset}"
    
    offline_epochs_list = []
    
    for run in runs:
        if run.name != run_name_pattern:
            continue
        
        try:
            seed = run.config.get('seed', None)
            if seed not in valid_seeds:
                continue
        except:
            continue
        
        # Get stabilization epochs performed
        history = run.history(keys=["stabilization_epochs_performed"], pandas=True)
        
        if "stabilization_epochs_performed" in history.columns:
            total_epochs = history["stabilization_epochs_performed"].sum()
            offline_epochs_list.append(total_epochs)
            print(f"    {run.name} (seed={seed}): {total_epochs:,.0f} offline epochs")
    
    return offline_epochs_list

def compute_gradient_steps_table(entity, project, environments, valid_seeds, dataset='expert',
                                  online_steps=300000, offline_pretrain_iql_calql=300000):
    """
    Compute total gradient steps for each algorithm and environment.
    
    Args:
        online_steps: Number of online environment steps (default: 300,000)
        offline_pretrain_iql_calql: Offline pretraining steps for IQL/CalQL (default: 300,000)
    """
    api = wandb.Api()
    
    print(f"\n{'='*100}")
    print("COMPUTATIONAL COST COMPARISON - GRADIENT STEPS")
    print(f"{'='*100}")
    print(f"Online steps: {online_steps:,}")
    print(f"IQL/CalQL offline pretraining: {offline_pretrain_iql_calql:,}")
    print(f"Dataset: {dataset}")
    print(f"{'='*100}\n")
    
    # Algorithm configurations
    # Format: (online_cost_per_step, offline_formula_description)
    algo_configs = {
        'RLPD': {
            'online_per_step': 201,  # 1 policy + 20 UTD × 10 Q
            'offline_total': 0,
            'description': '1π + 20×10Q per step'
        },
        'SAC': {
            'online_per_step': 3,  # 1 policy + 2 Q (UTD=1)
            'offline_total': 0,
            'description': '1π + 2Q per step'
        },
        'IQL': {
            'online_per_step': 4,  # 1 V + 2 Q + 1 policy
            'offline_total': offline_pretrain_iql_calql * 4,  # V + 2Q + π per offline step
            'description': '1V + 2Q + 1π per step'
        },
        'Cal-QL': {
            'online_per_step': 3,  # 2 Q + 1 policy
            'offline_total': offline_pretrain_iql_calql * 3,  # 2Q + π per offline step
            'description': '2Q + 1π per step'
        },
        'SPEQ': {
            'online_per_step': 3,  # 1 policy + 2 Q
            # Offline: every 10k steps, perform 75k epochs × 2 Q-networks
            'offline_total': (online_steps // 10000) * 75000 * 2,
            'description': '1π + 2Q + 75k×2 every 10k'
        },
        # PASPEQ will be computed dynamically from WandB
    }
    
    results = []
    
    for env in environments:
        print(f"\n{env}:")
        env_results = {'Environment': env.replace('-v5', '')}
        
        # Compute for fixed-formula algorithms
        for algo_name, config in algo_configs.items():
            online_cost = online_steps * config['online_per_step']
            offline_cost = config['offline_total']
            total = online_cost + offline_cost
            env_results[algo_name] = total
            print(f"  {algo_name:12s}: {total:>15,} (online: {online_cost:,}, offline: {offline_cost:,})")
        
        # Compute PASPEQ O2O from WandB (dynamic offline epochs)
        print(f"  Fetching PASPEQ O2O from WandB...")
        offline_epochs_list = get_paspeq_offline_epochs(api, entity, project, env, valid_seeds, dataset)
        
        if offline_epochs_list:
            mean_offline = np.mean(offline_epochs_list)
            std_offline = np.std(offline_epochs_list, ddof=1) if len(offline_epochs_list) > 1 else 0
            
            online_cost = online_steps * 3
            offline_cost_mean = mean_offline * 2  # 2 Q-networks per offline epoch
            offline_cost_std = std_offline * 2
            
            total_mean = online_cost + offline_cost_mean
            total_std = offline_cost_std
            
            env_results['PASPEQ_mean'] = total_mean
            env_results['PASPEQ_std'] = total_std
            env_results['PASPEQ_offline_epochs'] = mean_offline
            
            print(f"  {'PASPEQ O2O':12s}: {total_mean:>15,.0f} ± {total_std:,.0f} (offline epochs: {mean_offline:,.0f})")
        else:
            env_results['PASPEQ_mean'] = np.nan
            env_results['PASPEQ_std'] = np.nan
            env_results['PASPEQ_offline_epochs'] = np.nan
            print(f"  {'PASPEQ O2O':12s}: No data found")
        
        results.append(env_results)
    
    return pd.DataFrame(results)

def format_table(df, online_steps=300000):
    """Format the results into a publication-ready table."""
    
    # RLPD total (constant across all environments)
    rlpd_total = online_steps * 201
    
    print(f"\n\n{'='*140}")
    print("COMPUTATIONAL COST COMPARISON TABLE")
    print(f"{'='*140}")
    
    # Header
    header = f"{'Environment':<25} {'RLPD':>15} {'SAC':>15} {'IQL':>15} {'Cal-QL':>15} {'SPEQ':>15} {'PASPEQ O2O':>25} {'Reduction vs RLPD':>18}"
    print(header)
    print("-" * 140)
    
    # Data rows
    total_reductions = []
    
    for _, row in df.iterrows():
        env = row['Environment']
        rlpd = row['RLPD']
        sac = row['SAC']
        iql = row['IQL']
        calql = row['Cal-QL']
        speq = row['SPEQ']
        
        paspeq_mean = row['PASPEQ_mean']
        paspeq_std = row['PASPEQ_std']
        
        if not np.isnan(paspeq_mean):
            paspeq_str = f"{paspeq_mean:,.0f} ± {paspeq_std:,.0f}"
            reduction = (1 - paspeq_mean / rlpd) * 100
            reduction_str = f"{reduction:.1f}%"
            total_reductions.append(reduction)
        else:
            paspeq_str = "N/A"
            reduction_str = "N/A"
        
        print(f"{env:<25} {rlpd:>15,} {sac:>15,} {iql:>15,} {calql:>15,} {speq:>15,} {paspeq_str:>25} {reduction_str:>18}")
    
    print("-" * 140)
    
    # Averages
    avg_rlpd = df['RLPD'].mean()
    avg_sac = df['SAC'].mean()
    avg_iql = df['IQL'].mean()
    avg_calql = df['Cal-QL'].mean()
    avg_speq = df['SPEQ'].mean()
    avg_paspeq = df['PASPEQ_mean'].mean()
    avg_reduction = np.mean(total_reductions) if total_reductions else np.nan
    
    print(f"{'Average':<25} {avg_rlpd:>15,.0f} {avg_sac:>15,.0f} {avg_iql:>15,.0f} {avg_calql:>15,.0f} {avg_speq:>15,.0f} {avg_paspeq:>25,.0f} {avg_reduction:>17.1f}%")
    print(f"{'='*140}")
    
    # Summary comparison
    print(f"\n\nSUMMARY - Average Total Gradient Steps:")
    print(f"  RLPD:       {avg_rlpd:>15,.0f} (baseline)")
    print(f"  SAC:        {avg_sac:>15,.0f} ({(1 - avg_sac/avg_rlpd)*100:>6.1f}% reduction vs RLPD)")
    print(f"  IQL:        {avg_iql:>15,.0f} ({(1 - avg_iql/avg_rlpd)*100:>6.1f}% reduction vs RLPD)")
    print(f"  Cal-QL:     {avg_calql:>15,.0f} ({(1 - avg_calql/avg_rlpd)*100:>6.1f}% reduction vs RLPD)")
    print(f"  SPEQ:       {avg_speq:>15,.0f} ({(1 - avg_speq/avg_rlpd)*100:>6.1f}% reduction vs RLPD)")
    print(f"  PASPEQ O2O: {avg_paspeq:>15,.0f} ({avg_reduction:>6.1f}% reduction vs RLPD)")
    
    return df

def export_latex_table(df, output_path, online_steps=300000, offline_pretrain=300000):
    """Export results as LaTeX table with fixed costs in header and per-environment PASPEQ."""
    
    # Fixed costs (constant across all environments)
    rlpd_total = online_steps * 201
    sac_total = online_steps * 3
    iql_total = online_steps * 4 + offline_pretrain * 4
    calql_total = online_steps * 3 + offline_pretrain * 3
    speq_total = online_steps * 3 + (online_steps // 10000) * 75000 * 2
    
    latex = r"""\begin{table}[t]
\centering
\caption{Computational Cost Comparison: Total Gradient Steps}
\label{tab:gradient_steps}
\small
\begin{tabular}{lcccccc}
\toprule
"""
    
    # Header row 1: Fixed algorithm costs
    latex += r"\multicolumn{7}{l}{\textbf{Fixed Costs (constant across environments):}}" + "\n"
    latex += r"\\" + "\n"
    latex += f"\\multicolumn{{7}}{{l}}{{RLPD: {rlpd_total/1e6:.2f}M \\quad SAC: {sac_total/1e6:.2f}M \\quad IQL: {iql_total/1e6:.2f}M \\quad Cal-QL: {calql_total/1e6:.2f}M \\quad SPEQ: {speq_total/1e6:.2f}M}}" + "\n"
    latex += r"\\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Header row 2: Per-environment columns
    latex += r"\textbf{Environment} & \textbf{Offline Epochs} & \textbf{PASPEQ Total} & \textbf{vs RLPD} & \textbf{vs IQL} & \textbf{vs Cal-QL} & \textbf{vs SPEQ} \\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Data rows
    for _, row in df.iterrows():
        env = row['Environment']
        paspeq_mean = row['PASPEQ_mean']
        paspeq_std = row['PASPEQ_std']
        offline_epochs = row['PASPEQ_offline_epochs']
        
        if not np.isnan(paspeq_mean):
            offline_str = f"{offline_epochs/1e3:.0f}k"
            paspeq_str = f"{paspeq_mean/1e6:.2f}M $\\pm$ {paspeq_std/1e3:.0f}k"
            
            red_rlpd = (1 - paspeq_mean / rlpd_total) * 100
            red_iql = (1 - paspeq_mean / iql_total) * 100
            red_calql = (1 - paspeq_mean / calql_total) * 100
            red_speq = (1 - paspeq_mean / speq_total) * 100
            
            latex += f"{env} & {offline_str} & {paspeq_str} & {red_rlpd:.1f}\\% & {red_iql:.1f}\\% & {red_calql:.1f}\\% & {red_speq:.1f}\\% \\\\\n"
        else:
            latex += f"{env} & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n"
    
    latex += r"\midrule" + "\n"
    
    # Average row
    avg_paspeq = df['PASPEQ_mean'].mean()
    avg_offline = df['PASPEQ_offline_epochs'].mean()
    
    avg_red_rlpd = (1 - avg_paspeq / rlpd_total) * 100
    avg_red_iql = (1 - avg_paspeq / iql_total) * 100
    avg_red_calql = (1 - avg_paspeq / calql_total) * 100
    avg_red_speq = (1 - avg_paspeq / speq_total) * 100
    
    latex += f"\\textbf{{Average}} & {avg_offline/1e3:.0f}k & {avg_paspeq/1e6:.2f}M & {avg_red_rlpd:.1f}\\% & {avg_red_iql:.1f}\\% & {avg_red_calql:.1f}\\% & {avg_red_speq:.1f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="expert", choices=["expert", "medium", "simple"])
    parser.add_argument("--online-steps", type=int, default=300000, help="Online training steps")
    parser.add_argument("--offline-pretrain", type=int, default=300000, help="IQL/CalQL offline pretraining steps")
    parser.add_argument("--output-latex", type=str, default="gradient_steps_table.tex", help="LaTeX output file")
    args = parser.parse_args()
    
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    valid_seeds = {0, 42, 1234, 5678, 9876}
    
    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    if args.dataset == "simple":
        environments = simple_only_envs
    else:
        environments = all_envs
    
    print(f"Seeds: {sorted(valid_seeds)}")
    print(f"Environments: {len(environments)}")
    
    df = compute_gradient_steps_table(
        entity, project, environments, valid_seeds, args.dataset,
        online_steps=args.online_steps,
        offline_pretrain_iql_calql=args.offline_pretrain
    )
    
    format_table(df, online_steps=args.online_steps)
    export_latex_table(df, args.output_latex, online_steps=args.online_steps, offline_pretrain=args.offline_pretrain)