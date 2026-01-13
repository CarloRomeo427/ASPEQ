import wandb
from collections import defaultdict

def check_run_completeness(entity, project, algorithms, environments, seeds, datasets,
                          metric_name="EvalReward", expected_points=300):
    api = wandb.Api()
    print(f"Fetching runs from {entity}/{project}...")
    all_runs = api.runs(f"{entity}/{project}")
    
    # results[algo][env][dataset][seed]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for run in all_runs:
        name = run.name
        
        # Determine algorithm
        algo = None
        for a in algorithms:
            if name.startswith(a + "_"):
                algo = a
                break
        if not algo:
            continue
        
        remainder = name[len(algo) + 1:]  # after "algo_"
        
        # Determine dataset suffix (check from end)
        dataset = "expert"
        if remainder.endswith("_medium"):
            dataset = "medium"
            remainder = remainder[:-7]  # remove "_medium"
        elif remainder.endswith("_simple"):
            dataset = "simple"
            remainder = remainder[:-7]  # remove "_simple"
        
        # remainder should now be the environment name (e.g., "Humanoid-v5")
        matched_env = None
        for e in environments:
            if remainder == e:
                matched_env = e
                break
        if not matched_env:
            continue
        
        # Validate dataset applicability
        simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
        if dataset == "simple" and matched_env not in simple_only_envs:
            continue
        
        try:
            seed = run.config.get('seed', None)
            if seed is None or seed not in seeds:
                continue
        except:
            continue
        
        state = run.state
        history = run.history(keys=[metric_name], pandas=True)
        eval_points = history[metric_name].dropna().shape[0] if metric_name in history.columns else 0
        
        results[algo][matched_env][dataset][seed] = {
            'run_name': run.name,
            'state': state,
            'eval_points': eval_points,
            'complete': eval_points >= expected_points
        }
    
    return results, expected_points

def print_report(results, expected_points, algorithms, environments, seeds, datasets):
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    
    print("\n" + "="*80)
    print("RUN COMPLETENESS REPORT")
    print("="*80)
    print(f"Expected points per run: {expected_points}")
    print(f"Seeds: {seeds}\n")
    
    for algo in algorithms:
        print(f"\n{'─'*80}")
        print(f"Algorithm: {algo.upper()}")
        print(f"{'─'*80}")
        
        for env in environments:
            applicable_datasets = datasets if env in simple_only_envs else [d for d in datasets if d != "simple"]
            
            for dataset in applicable_datasets:
                data = results[algo].get(env, {}).get(dataset, {})
                
                missing = [s for s in seeds if s not in data]
                complete = [(s, data[s]) for s in seeds if s in data and data[s]['complete']]
                running = [(s, data[s]) for s in seeds if s in data and data[s]['state'] == 'running']
                crashed = [(s, data[s]) for s in seeds if s in data and data[s]['state'] in ['crashed', 'failed']]
                incomplete = [(s, data[s]) for s in seeds if s in data and not data[s]['complete'] and data[s]['state'] not in ['running', 'crashed', 'failed']]
                
                label = f"{env} [{dataset}]"
                print(f"\n  {label}:")
                print(f"    Complete: {len(complete)}/{len(seeds)}")
                
                if missing:
                    print(f"    ⚠️  Missing: {missing}")
                if incomplete:
                    print(f"    ⚠️  Incomplete: {[(s, r['eval_points']) for s, r in incomplete]}")
                if running:
                    print(f"    🔄 Running: {[(s, r['eval_points']) for s, r in running]}")
                if crashed:
                    print(f"    ❌ Crashed: {[(s, r['eval_points']) for s, r in crashed]}")
    
    print("\n" + "="*80)

def generate_missing_runs_list(results, expected_points, algorithms, environments, seeds, datasets):
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    missing_runs = []
    
    for algo in algorithms:
        for env in environments:
            applicable_datasets = datasets if env in simple_only_envs else [d for d in datasets if d != "simple"]
            
            for dataset in applicable_datasets:
                data = results[algo].get(env, {}).get(dataset, {})
                
                for seed in seeds:
                    suffix = "" if dataset == "expert" else f"_{dataset}"
                    if seed not in data:
                        missing_runs.append({
                            'algo': algo, 'env': env, 'dataset': dataset, 'seed': seed,
                            'status': 'NOT_STARTED', 'run_name': f"{algo}_{env}{suffix}"
                        })
                    elif not data[seed]['complete'] and data[seed]['state'] != 'running':
                        missing_runs.append({
                            'algo': algo, 'env': env, 'dataset': dataset, 'seed': seed,
                            'status': f"{data[seed]['state'].upper()}_{data[seed]['eval_points']}pts",
                            'run_name': data[seed]['run_name']
                        })
    
    return missing_runs

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    algorithms = ["paspeq_o2o", "rlpd"]
    environments = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                    "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    seeds = [0, 42, 1234, 5678, 9876, 777, 24680, 13579, 31415, 27182]
    datasets = ["expert", "medium", "simple"]
    
    results, expected_points = check_run_completeness(
        entity, project, algorithms, environments, seeds, datasets,
        metric_name="EvalReward", expected_points=300
    )
    
    print_report(results, expected_points, algorithms, environments, seeds, datasets)
    
    missing = generate_missing_runs_list(results, expected_points, algorithms, environments, seeds, datasets)
    
    if missing:
        print("\nRUNS TO START/RESTART:")
        print("="*80)
        for run in missing:
            print(f"  {run['algo']:12} {run['env']:25} {run['dataset']:8} seed={run['seed']:<5} [{run['status']}]")
        print(f"\nTotal: {len(missing)} runs need attention")
    else:
        print("\n✅ All runs complete!")