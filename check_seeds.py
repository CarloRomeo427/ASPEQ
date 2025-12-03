import wandb
import pandas as pd
from collections import defaultdict

def check_run_completeness(entity, project, algorithms, environments, seeds,
                          metric_name="EvalReward", expected_points=300):
    """
    Check which runs are missing or incomplete across algorithms and environments.
    
    Args:
        entity: WandB entity
        project: WandB project name
        algorithms: List of algorithm names (e.g., ["aspeq", "speq"])
        environments: List of environment names (e.g., ["Humanoid-v5", "Ant-v5"])
        seeds: List of seed values to check for (e.g., [0, 21, 42])
        metric_name: Metric to check for completeness
        expected_points: Expected number of evaluation points
    
    Returns:
        dict: Report of missing/incomplete runs
    """
    api = wandb.Api()
    
    # Get all runs from project
    print(f"Fetching runs from {entity}/{project}...")
    all_runs = api.runs(f"{entity}/{project}")
    
    # Organize runs by algorithm, environment, and seed
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for run in all_runs:
        # Parse run name: algo_env-v5 (and extract seed from config or name)
        parts = run.name.split("_")
        if len(parts) < 2:
            continue
        
        algo = parts[0]
        env_part = "_".join(parts[1:])
        
        # Check if this run matches our algorithms
        if algo not in algorithms:
            continue
        
        # Find matching environment
        matched_env = None
        for e in environments:
            if env_part.startswith(e):
                matched_env = e
                break
        
        if not matched_env:
            continue
        
        # Get seed from run config
        try:
            seed = run.config.get('seed', None)
            if seed is None:
                continue
        except:
            continue
        
        # Get run state and data
        state = run.state  # running, finished, crashed, failed
        
        # Count evaluation points
        history = run.history(keys=[metric_name], pandas=True)
        
        if metric_name in history.columns:
            eval_points = history[metric_name].dropna().shape[0]
        else:
            eval_points = 0
        
        results[algo][matched_env][seed] = {
            'run_name': run.name,
            'state': state,
            'eval_points': eval_points,
            'complete': eval_points >= expected_points
        }
    
    return results, expected_points

def print_report(results, expected_points, algorithms, environments, seeds):
    """Print formatted report of missing/incomplete runs."""
    
    print("\n" + "="*80)
    print("RUN COMPLETENESS REPORT")
    print("="*80)
    print(f"Expected points per run: {expected_points}")
    print(f"Expected seeds: {seeds}\n")
    
    for algo in algorithms:
        print(f"\n{'─'*80}")
        print(f"Algorithm: {algo.upper()}")
        print(f"{'─'*80}")
        
        for env in environments:
            env_data = results[algo].get(env, {})
            
            print(f"\n  {env}:")
            
            missing_seeds = []
            incomplete_runs = []
            running_runs = []
            crashed_runs = []
            complete_runs = []
            
            for seed in seeds:
                if seed not in env_data:
                    missing_seeds.append(seed)
                else:
                    run_info = env_data[seed]
                    if run_info['complete']:
                        complete_runs.append((seed, run_info))
                    elif run_info['state'] == 'running':
                        running_runs.append((seed, run_info))
                    elif run_info['state'] in ['crashed', 'failed']:
                        crashed_runs.append((seed, run_info))
                    else:
                        incomplete_runs.append((seed, run_info))
            
            # Print summary
            print(f"    Complete: {len(complete_runs)}/{len(seeds)} seeds")
            
            if missing_seeds:
                print(f"    ⚠️  Missing seeds (not started): {missing_seeds}")
            
            if incomplete_runs:
                print(f"    ⚠️  Incomplete seeds (finished but < {expected_points} points):")
                for seed, r in incomplete_runs:
                    print(f"       - Seed {seed} ({r['run_name']}): {r['eval_points']} points")
            
            if running_runs:
                print(f"    🔄 Running seeds:")
                for seed, r in running_runs:
                    print(f"       - Seed {seed} ({r['run_name']}): {r['eval_points']} points so far")
            
            if crashed_runs:
                print(f"    ❌ Crashed/Failed seeds:")
                for seed, r in crashed_runs:
                    print(f"       - Seed {seed} ({r['run_name']}): {r['eval_points']} points")
    
    print("\n" + "="*80)

def generate_missing_runs_list(results, expected_points, algorithms, environments, seeds):
    """Generate a list of runs that need to be started or restarted."""
    
    missing_runs = []
    
    for algo in algorithms:
        for env in environments:
            env_data = results[algo].get(env, {})
            
            for seed in seeds:
                if seed not in env_data:
                    # Seed is completely missing - need to start
                    missing_runs.append({
                        'algo': algo,
                        'env': env,
                        'seed': seed,
                        'status': 'NOT_STARTED',
                        'run_name': f"{algo}_{env}_seed{seed}"
                    })
                else:
                    run_info = env_data[seed]
                    if not run_info['complete'] and run_info['state'] != 'running':
                        # Run exists but is incomplete/crashed - need to restart
                        missing_runs.append({
                            'algo': algo,
                            'env': env,
                            'seed': seed,
                            'status': f"{run_info['state'].upper()}_{run_info['eval_points']}pts",
                            'run_name': run_info['run_name']
                        })
    
    return missing_runs

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    algorithms = ["aspeq", "speq"]
    environments = [
        "Humanoid-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", 
        "Hopper-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5", 
        "Reacher-v5", "Swimmer-v5", "Pusher-v5"
    ]
    seeds = [0, 42, 1234, 5678]  # List of expected seeds
    
    # Check completeness
    results, expected_points = check_run_completeness(
        entity, project, algorithms, environments, seeds,
        metric_name="EvalReward",
        expected_points=300
    )
    
    # Print report
    print_report(results, expected_points, algorithms, environments, seeds)
    
    # Generate list of missing runs
    missing = generate_missing_runs_list(results, expected_points, algorithms, environments, seeds)
    
    if missing:
        print("\n" + "="*80)
        print("RUNS TO START/RESTART:")
        print("="*80)
        for run in missing:
            print(f"  {run['algo']:10} {run['env']:30} seed={run['seed']:3}  [{run['status']}]")
        print(f"\nTotal: {len(missing)} runs need attention")
    else:
        print("\n✅ All runs complete!")