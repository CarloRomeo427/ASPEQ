import wandb
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def exponential_moving_average(data, alpha=0.05):
    if len(data) == 0:
        return data
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def auto_label(algo):
    direct = {
        'rlpd': 'RLPD',
        'iql': 'IQL',
        'calql': 'Cal-QL',
        'sacfd': 'SACfD',
        'sac': 'SAC',
    }
    if algo in direct:
        return direct[algo]
    label = algo
    label = label.replace('_o2o', ' O2O')
    label = label.replace('_pi', ' (π)')
    label = label.replace('_td', ' (TD)')
    label = label.replace('pct10', '10%')
    label = label.replace('pct20', '20%')
    label = label.replace('pct', '%')
    label = label.replace('valpat', 'pat=')
    label = label.replace('_', ' ')
    while '  ' in label:
        label = label.replace('  ', ' ')
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

def get_eval_reward_for_run(api, entity, project, run_name_patterns, metric_name="EvalReward", valid_seeds=None):
    if isinstance(run_name_patterns, str):
        run_name_patterns = [run_name_patterns]
    runs = api.runs(f"{entity}/{project}")
    all_seeds_data = []
    for run in runs:
        if run.name in run_name_patterns:
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
                    print(f"    Loaded {run.name} (seed={run.config.get('seed')}): {len(history)} points")
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

NO_EXPERT_SUFFIX_ALGOS = {'rlpd'}

def get_env_variants(env):
    variants = [env]
    cap_version = env.capitalize()
    if cap_version not in variants:
        variants.append(cap_version)
    lower_version = env.lower()
    if lower_version not in variants:
        variants.append(lower_version)
    return variants

def parse_algo_name(algo):
    if '_valpat' in algo:
        parts = algo.rsplit('_valpat', 1)
        return parts[0], f"valpat{parts[1]}"
    return algo, None

def build_run_name(algo, env, dataset):
    base_algo, suffix = parse_algo_name(algo)
    if base_algo in NO_EXPERT_SUFFIX_ALGOS and dataset == "expert":
        run_name = f"{base_algo}_{env}"
    else:
        run_name = f"{base_algo}_{env}_{dataset}"
    if suffix:
        run_name = f"{run_name}_{suffix}"
    return run_name

def build_run_name_variants(algo, env, dataset):
    variants = []
    for env_variant in get_env_variants(env):
        variants.append(build_run_name(algo, env_variant, dataset))
    return variants

def collect_all_data(api, entity, project, envs, algorithms, valid_seeds, dataset, metric_name="EvalReward"):
    algo_env_data = {algo: {} for algo in algorithms}
    for env in envs:
        print(f"\n{env}:")
        for algo in algorithms:
            run_patterns = build_run_name_variants(algo, env, dataset)
            print(f"  Looking for {algo}: {run_patterns[0]}...")
            all_seeds_data = get_eval_reward_for_run(api, entity, project, run_patterns, metric_name, valid_seeds)
            if not all_seeds_data:
                print(f"    No data found")
                algo_env_data[algo][env] = None
                continue
            steps, mean, std = compute_mean_std_across_seeds(all_seeds_data, metric_name)
            algo_env_data[algo][env] = (steps, mean, std, len(all_seeds_data))
            print(f"    Found {len(all_seeds_data)} seeds, final: {mean[-1]:.2f} ± {std[-1]:.2f}")
    return algo_env_data

def plot_single_env(algo_env_data, env, algorithms, ema_alpha=0.05, std_alpha=0.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    label_map = {algo: auto_label(algo) for algo in algorithms}
    for algo in algorithms:
        data = algo_env_data[algo].get(env)
        if data is None:
            continue
        steps, mean, std, n_seeds = data
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        color = color_map[algo]
        label = label_map[algo]
        ax.plot(steps, mean_ema, label=label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color)
    env_display = env.replace("-v5", "")
    ax.set_title(f"{env_display}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_ylabel("Eval Reward", fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_normalizing_boundaries(path="normalizing_boundaries.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run compute_normalizing_boundaries.py first.")
    with open(path) as f:
        return json.load(f)

def normalize_score(raw_score, env_min, env_max):
    denom = env_max - env_min
    if abs(denom) < 1e-8:
        if isinstance(raw_score, np.ndarray):
            return np.zeros_like(raw_score)
        return 0.0
    return (raw_score - env_min) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated (normalized) computation & plotting
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregated_performance(algo_env_data, algorithms, envs, boundaries=None, n_points=300):
    aggregated = {}
    for algo in algorithms:
        env_curves = []
        for env in envs:
            data = algo_env_data[algo].get(env)
            if data is None:
                continue
            steps, mean, std, n_seeds = data
            if boundaries is not None and env in boundaries:
                env_min = boundaries[env]["min"]
                env_max = boundaries[env]["max"]
                mean = normalize_score(mean, env_min, env_max)
                std = std / (env_max - env_min) if abs(env_max - env_min) > 1e-8 else std
            env_curves.append((steps, mean))
        if not env_curves:
            aggregated[algo] = None
            continue
        max_step = max(c[0].max() for c in env_curves)
        common_steps = np.linspace(0, max_step, n_points)
        interpolated = [np.interp(common_steps, steps, mean) for steps, mean in env_curves]
        stacked = np.array(interpolated)
        agg_mean = np.mean(stacked, axis=0)
        agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
        aggregated[algo] = (common_steps, agg_mean, agg_std, len(env_curves))
    return aggregated

def plot_aggregated(aggregated_data, algorithms, dataset, ema_alpha=0.05, std_alpha=0.1,
                    ax=None, normalized=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    label_map = {algo: auto_label(algo) for algo in algorithms}
    for algo in algorithms:
        data = aggregated_data.get(algo)
        if data is None:
            continue
        steps, mean, std, n_envs = data
        mean_ema = exponential_moving_average(mean, alpha=ema_alpha)
        std_ema = exponential_moving_average(std, alpha=ema_alpha)
        color = color_map[algo]
        label = f"{label_map[algo]} ({n_envs} envs)"
        ax.plot(steps, mean_ema, label=label, linewidth=2, color=color)
        ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=std_alpha, color=color)
    ylabel = "Normalized Score" if normalized else "Eval Reward (avg across envs)"
    ax.set_title(f"Aggregated Performance - {dataset.upper()} Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

def plot_combined_envs(algo_env_data, envs, algorithms, dataset, ema_alpha, std_alpha):
    n_envs = len(envs)
    n_cols = min(5, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    for idx, env in enumerate(envs):
        plot_single_env(algo_env_data, env, algorithms, ema_alpha, std_alpha, ax=axes[idx])
    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f"Performance Comparison - {dataset.upper()} Dataset", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def print_summary_statistics(algo_env_data, algorithms, envs, boundaries=None):
    label_map = {algo: auto_label(algo) for algo in algorithms}
    header = "SUMMARY STATISTICS (Final Normalized Scores)" if boundaries else "SUMMARY STATISTICS (Final Scores)"
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    for algo in algorithms:
        final_scores = []
        for env in envs:
            data = algo_env_data[algo].get(env)
            if data is not None:
                steps, mean, std, n_seeds = data
                score = mean[-1]
                if boundaries and env in boundaries:
                    score = normalize_score(score, boundaries[env]["min"], boundaries[env]["max"])
                final_scores.append(score)
        label = label_map[algo]
        if final_scores:
            spread = np.std(final_scores, ddof=1) if len(final_scores) > 1 else 0
            print(f"{label:30s}: {np.mean(final_scores):8.4f} ± {spread:6.4f}  (n={len(final_scores)} envs)")
        else:
            print(f"{label:30s}: No data")


# ─────────────────────────────────────────────────────────────────────────────
# Table generation (--table mode)
# ─────────────────────────────────────────────────────────────────────────────

def collect_table_scores(all_datasets_data, algorithms, boundaries):
    """
    Build structured dict of final normalized scores.
    scores[dataset][env][algo] = (norm_mean, norm_std, n_seeds)
    Uses last 10 eval points for stability.
    """
    scores = {}
    for dataset, (algo_env_data, envs) in all_datasets_data.items():
        scores[dataset] = {}
        for env in envs:
            scores[dataset][env] = {}
            for algo in algorithms:
                data = algo_env_data[algo].get(env)
                if data is None:
                    scores[dataset][env][algo] = None
                    continue
                steps, mean, std, n_seeds = data
                window = min(10, len(mean))
                final_mean = np.mean(mean[-window:])
                final_std = np.mean(std[-window:])
                if boundaries and env in boundaries:
                    env_min = boundaries[env]["min"]
                    env_max = boundaries[env]["max"]
                    final_mean = normalize_score(final_mean, env_min, env_max)
                    denom = env_max - env_min
                    final_std = final_std / denom if abs(denom) > 1e-8 else final_std
                scores[dataset][env][algo] = (final_mean, final_std, n_seeds)
    return scores


def format_cell(mean, std, is_best=False, fmt=".1f"):
    """Format a table cell as mean±std, bold if best."""
    if mean is None:
        return "—"
    text = f"{mean:{fmt}} \\pm {std:{fmt}}"
    if is_best:
        return f"$\\mathbf{{{text}}}$"
    return f"${text}$"


def generate_latex_table(scores, algorithms, boundaries, our_algo="faspeq_pct10_pi"):
    """
    Generate LaTeX table(s): one per dataset.
    Rows = environments + aggregate. Columns = algorithms.
    Best score per row is bolded. Scores displayed as percentages (×100).
    """
    label_map = {a: auto_label(a) for a in algorithms}

    # LaTeX-safe column headers
    latex_labels = {}
    for a in algorithms:
        lbl = label_map[a]
        lbl = lbl.replace("π", "$\\pi$")
        if a == our_algo:
            latex_labels[a] = f"\\textbf{{{lbl} (Ours)}}"
        else:
            latex_labels[a] = lbl

    all_latex = []

    for dataset, env_dict in scores.items():
        envs = list(env_dict.keys())
        n_algos = len(algorithms)
        col_spec = "l" + "c" * n_algos

        lines = []
        lines.append(f"% === {dataset.upper()} Dataset ===")
        lines.append(f"\\begin{{table*}}[ht]")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{Normalized scores (\\%) on \\textbf{{{dataset}}} dataset "
                      f"(mean $\\pm$ std across seeds). Best per environment in bold.}}")
        lines.append(f"\\label{{tab:results_{dataset}}}")
        lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(f"\\toprule")

        header = "Environment & " + " & ".join(latex_labels[a] for a in algorithms) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        algo_all_scores = {a: [] for a in algorithms}

        for env in envs:
            env_display = env.replace("-v5", "")

            # Find best mean
            means = {}
            for a in algorithms:
                entry = env_dict[env].get(a)
                if entry is not None:
                    means[a] = entry[0]
            best_algo = max(means, key=means.get) if means else None

            cells = []
            for a in algorithms:
                entry = env_dict[env].get(a)
                if entry is None:
                    cells.append("—")
                else:
                    m, s, n = entry
                    is_best = (a == best_algo)
                    cells.append(format_cell(m * 100, s * 100, is_best=is_best))
                    algo_all_scores[a].append(m)
            row = f"{env_display} & " + " & ".join(cells) + " \\\\"
            lines.append(row)

        # Aggregate row
        lines.append("\\midrule")
        agg_means = {}
        agg_stds = {}
        for a in algorithms:
            vals = algo_all_scores[a]
            if vals:
                agg_means[a] = np.mean(vals)
                agg_stds[a] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            else:
                agg_means[a] = None

        best_agg = max((a for a in algorithms if agg_means.get(a) is not None),
                       key=lambda a: agg_means[a], default=None)

        agg_cells = []
        for a in algorithms:
            if agg_means.get(a) is None:
                agg_cells.append("—")
            else:
                is_best = (a == best_agg)
                agg_cells.append(format_cell(agg_means[a] * 100, agg_stds[a] * 100, is_best=is_best))
        lines.append(f"\\textbf{{Average}} & " + " & ".join(agg_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")
        lines.append("\\end{table*}")

        all_latex.append("\n".join(lines))

    return "\n\n".join(all_latex)


def generate_console_table(scores, algorithms):
    """Print a readable console table."""
    label_map = {a: auto_label(a) for a in algorithms}
    col_w = 20
    env_w = 28

    for dataset, env_dict in scores.items():
        envs = list(env_dict.keys())

        print(f"\n{'='*140}")
        print(f"  {dataset.upper()} DATASET — Normalized Scores (×100)")
        print(f"{'='*140}")

        header = f"{'Environment':<{env_w}}" + "".join(f"{label_map[a]:>{col_w}}" for a in algorithms)
        print(header)
        print("-" * (env_w + col_w * len(algorithms)))

        algo_all = {a: [] for a in algorithms}

        for env in envs:
            env_short = env.replace("-v5", "")
            row = f"{env_short:<{env_w}}"

            means = {}
            for a in algorithms:
                entry = env_dict[env].get(a)
                if entry is not None:
                    means[a] = entry[0]
            best = max(means, key=means.get) if means else None

            for a in algorithms:
                entry = env_dict[env].get(a)
                if entry is None:
                    row += f"{'—':>{col_w}}"
                else:
                    m, s, n = entry
                    marker = " *" if a == best else "  "
                    cell = f"{m*100:6.1f} ± {s*100:5.1f}{marker}"
                    row += f"{cell:>{col_w}}"
                    algo_all[a].append(m)
            print(row)

        print("-" * (env_w + col_w * len(algorithms)))

        # Aggregate
        row = f"{'AVERAGE':<{env_w}}"
        agg_means = {}
        for a in algorithms:
            vals = algo_all[a]
            if vals:
                agg_means[a] = np.mean(vals)
        best_agg = max((a for a in algorithms if a in agg_means),
                       key=lambda a: agg_means[a], default=None)

        for a in algorithms:
            vals = algo_all[a]
            if vals:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                marker = " *" if a == best_agg else "  "
                cell = f"{m*100:6.1f} ± {s*100:5.1f}{marker}"
                row += f"{cell:>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)
        print(f"{'='*140}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=["expert", "medium", "simple"])
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--no-aggregated", action="store_true")
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--std-alpha", type=float, default=0.05)
    parser.add_argument("--boundaries", type=str, default="normalizing_boundaries.json")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--table", action="store_true",
                        help="Generate per-(env, dataset, algo) comparison table (console + LaTeX)")
    parser.add_argument("--our-algo", type=str, default="faspeq_pct10_pi",
                        help="Algorithm to highlight as 'Ours' in LaTeX table")
    args = parser.parse_args()

    entity = "carlo-romeo-alt427"
    project = "SPEQ"
    valid_seeds = {0, 42, 1234, 5678, 9876}

    # ============================================================
    algorithms = [
        "speq_o2o",
        "rlpd",
        "calql",
        "sacfd",
        "faspeq_pct10_pi",
        # "paspeq_o2o"
    ]
    # ============================================================

    all_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                "InvertedPendulum-v5", "InvertedDoublePendulum-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
    simple_only_envs = ["Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]

    boundaries = None
    if not args.no_normalize:
        try:
            boundaries = load_normalizing_boundaries(args.boundaries)
            print(f"Loaded normalizing boundaries from {args.boundaries}")
        except FileNotFoundError:
            print(f"WARNING: {args.boundaries} not found — will use raw rewards.")

    if args.dataset:
        datasets_to_plot = [args.dataset]
    else:
        datasets_to_plot = ["expert", "medium", "simple"]

    api = wandb.Api()

    print("=" * 80)
    print("ALGORITHMS:")
    for algo in algorithms:
        print(f"  {algo} -> {auto_label(algo)}")
    print("=" * 80)

    os.makedirs("Plots", exist_ok=True)
    all_datasets_data = {}

    for dataset in datasets_to_plot:
        envs = simple_only_envs if dataset == "simple" else all_envs
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}")
        algo_env_data = collect_all_data(api, entity, project, envs, algorithms, valid_seeds, dataset)
        all_datasets_data[dataset] = (algo_env_data, envs)

    # ─── TABLE MODE ─────────────────────────────────────────────────────
    if args.table:
        scores = collect_table_scores(all_datasets_data, algorithms, boundaries)

        # Console output
        generate_console_table(scores, algorithms)

        # LaTeX output
        latex = generate_latex_table(scores, algorithms, boundaries, our_algo=args.our_algo)

        latex_path = "Plots/results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {latex_path}")
        print("\n" + "=" * 80)
        print("LATEX OUTPUT:")
        print("=" * 80)
        print(latex)

        print("\nDONE (table mode)")
        exit(0)

    # ─── PLOT MODE (original behaviour) ─────────────────────────────────
    for dataset in datasets_to_plot:
        algo_env_data, envs = all_datasets_data[dataset]

        if args.individual:
            for env in envs:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_single_env(algo_env_data, env, algorithms, args.ema_alpha, args.std_alpha, ax=ax)
                plt.tight_layout()
                plt.savefig(f"Plots/{env}_{dataset}.png", dpi=300, bbox_inches="tight")
                plt.close()
        else:
            fig = plot_combined_envs(algo_env_data, envs, algorithms, dataset, args.ema_alpha, args.std_alpha)
            plt.savefig(f"Plots/combined_{dataset}.png", dpi=300, bbox_inches="tight")
            plt.close()

        if not args.no_aggregated:
            aggregated = compute_aggregated_performance(
                algo_env_data, algorithms, envs, boundaries=boundaries
            )
            fig, ax = plt.subplots(figsize=(12, 7))
            plot_aggregated(aggregated, algorithms, dataset, args.ema_alpha,
                            args.std_alpha * 2, ax=ax, normalized=boundaries is not None)
            plt.tight_layout()
            plt.savefig(f"Plots/aggregated_{dataset}.png", dpi=300, bbox_inches="tight")
            plt.close()

        print_summary_statistics(algo_env_data, algorithms, envs, boundaries)

    # Cross-dataset aggregated plot
    if len(datasets_to_plot) == 3 and not args.no_aggregated:
        print(f"\n{'='*80}")
        print("AGGREGATED ACROSS ALL DATASETS")
        print(f"{'='*80}")

        combined_aggregated = {}
        for algo in algorithms:
            all_curves = []
            for dataset in datasets_to_plot:
                algo_env_data, envs = all_datasets_data[dataset]
                for env in envs:
                    data = algo_env_data[algo].get(env)
                    if data is None:
                        continue
                    steps, mean, std, n_seeds = data
                    if boundaries is not None and env in boundaries:
                        mean = normalize_score(mean, boundaries[env]["min"], boundaries[env]["max"])
                    all_curves.append((steps, mean))
            if not all_curves:
                combined_aggregated[algo] = None
                continue
            max_step = max(c[0].max() for c in all_curves)
            common_steps = np.linspace(0, max_step, 300)
            interpolated = [np.interp(common_steps, s, m) for s, m in all_curves]
            stacked = np.array(interpolated)
            agg_mean = np.mean(stacked, axis=0)
            agg_std = np.std(stacked, axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(agg_mean)
            combined_aggregated[algo] = (common_steps, agg_mean, agg_std, len(all_curves))
            print(f"  {auto_label(algo)}: {len(all_curves)} env-dataset pairs")

        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.cm.tab10
        color_map = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
        label_map = {algo: auto_label(algo) for algo in algorithms}
        for algo in algorithms:
            data = combined_aggregated.get(algo)
            if data is None:
                continue
            steps, mean, std, n_curves = data
            mean_ema = exponential_moving_average(mean, alpha=args.ema_alpha)
            std_ema = exponential_moving_average(std, alpha=args.ema_alpha)
            color = color_map[algo]
            ax.plot(steps, mean_ema, label=f"{label_map[algo]} ({n_curves})", linewidth=2, color=color)
            ax.fill_between(steps, mean_ema - std_ema, mean_ema + std_ema, alpha=0.1, color=color)

        ylabel = "Normalized Score" if boundaries else "Eval Reward (avg)"
        ax.set_title("Aggregated Performance - ALL DATASETS", fontsize=14, fontweight='bold')
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("Plots/aggregated_all_datasets.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nDONE")