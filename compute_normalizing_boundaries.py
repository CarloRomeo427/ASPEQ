"""
Compute normalizing boundaries for MuJoCo environments.

For each environment:
  - min: average episode return of a uniform-random policy (across 5 seeds)
  - max: average episode return from the Minari expert dataset

Output: JSON file "normalizing_boundaries.json"
"""

import json
import argparse
import numpy as np
import gymnasium as gym
import minari


ENVS = {
    "Hopper-v5":                   ("hopper",                  "mujoco/hopper/expert-v0"),
    "HalfCheetah-v5":              ("halfcheetah",             "mujoco/halfcheetah/expert-v0"),
    "Walker2d-v5":                 ("walker2d",                "mujoco/walker2d/expert-v0"),
    "Ant-v5":                      ("ant",                     "mujoco/ant/expert-v0"),
    "Swimmer-v5":                  ("swimmer",                 "mujoco/swimmer/expert-v0"),
    "Humanoid-v5":                 ("humanoid",                "mujoco/humanoid/expert-v0"),
    "InvertedPendulum-v5":         ("invertedpendulum",        "mujoco/invertedpendulum/expert-v0"),
    "InvertedDoublePendulum-v5":   ("inverteddoublependulum",  "mujoco/inverteddoublependulum/expert-v0"),
    "Pusher-v5":                   ("pusher",                  "mujoco/pusher/expert-v0"),
    "Reacher-v5":                  ("reacher",                 "mujoco/reacher/expert-v0"),
}

SEEDS = [0, 42, 1234, 5678, 9876]
NUM_EVAL_EPISODES = 100  # per seed for random policy


def compute_random_min(env_id: str, seeds: list, num_episodes: int) -> float:
    """Run uniform-random policy, return average episode return across all seeds."""
    all_returns = []

    for seed in seeds:
        env = gym.make(env_id)
        np.random.seed(seed)

        for _ in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            ep_ret = 0.0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward
                done = terminated or truncated
            all_returns.append(ep_ret)

        env.close()

    return float(np.mean(all_returns))


def compute_expert_max(dataset_name: str) -> float:
    """Load Minari expert dataset, return average episode return."""
    try:
        dataset = minari.load_dataset(dataset_name)
    except Exception:
        print(f"  Downloading {dataset_name}...")
        minari.download_dataset(dataset_name)
        dataset = minari.load_dataset(dataset_name)

    episode_returns = []
    for episode in dataset.iterate_episodes():
        episode_returns.append(float(np.sum(episode.rewards)))

    return float(np.mean(episode_returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="normalizing_boundaries.json")
    parser.add_argument("--num-episodes", type=int, default=NUM_EVAL_EPISODES,
                        help="Episodes per seed for random policy evaluation")
    args = parser.parse_args()

    boundaries = {}

    for env_id, (canonical, expert_dataset) in ENVS.items():
        print(f"\n{'='*60}")
        print(f"Environment: {env_id}")
        print(f"{'='*60}")

        # --- Random (min) ---
        print(f"  Computing random baseline ({len(SEEDS)} seeds × {args.num_episodes} episodes)...")
        rand_mean = compute_random_min(env_id, SEEDS, args.num_episodes)
        print(f"  Random mean return: {rand_mean:.4f}")

        # --- Expert (max) ---
        print(f"  Computing expert baseline from {expert_dataset}...")
        expert_mean = compute_expert_max(expert_dataset)
        print(f"  Expert mean return: {expert_mean:.4f}")

        boundaries[env_id] = {"min": rand_mean, "max": expert_mean}

    # Save
    with open(args.output, "w") as f:
        json.dump(boundaries, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(json.dumps(boundaries, indent=2))


if __name__ == "__main__":
    main()
