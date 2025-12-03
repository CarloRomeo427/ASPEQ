import wandb
import pandas as pd
import numpy as np

def sum_metric_across_seeds(entity, project, run_name_pattern, metric_name="stabilization_epochs_performed"):
    """
    Sum metric values across all seeds and return mean ± std.
    
    Args:
        entity: WandB entity/username
        project: WandB project name
        run_name_pattern: Base run name (will match all runs starting with this)
        metric_name: Name of metric to sum
    
    Returns:
        tuple: (mean, std, list of individual sums)
    """
    api = wandb.Api()
    
    # Get all runs matching the pattern
    runs = api.runs(f"{entity}/{project}")
    
    sums = []
    
    for run in runs:
        # Match runs that start with the pattern
        if run.name.startswith(run_name_pattern):
            history = run.history(keys=[metric_name], pandas=True)
            
            if metric_name in history.columns:
                total = history[metric_name].sum()
                sums.append(total)
                print(f"{run.name}: {total}")
            else:
                print(f"{run.name}: Metric '{metric_name}' not found")
    
    if not sums:
        print(f"No runs found matching pattern '{run_name_pattern}'")
        return None, None, []
    
    mean = np.mean(sums)
    std = np.std(sums, ddof=1) if len(sums) > 1 else 0
    
    return mean, std, sums

if __name__ == "__main__":
    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    
    mean, std, individual_sums = sum_metric_across_seeds(entity, project, run_name_pattern="aspeq_Pusher-v5")
    
    print(f"Individual sums: {individual_sums}")
    print(f"Mean: {mean:.2f}")
    print(f"Std: {std:.2f}")
    print(f"Result: {mean:.2f} ± {std:.2f}")