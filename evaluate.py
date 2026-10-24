"""Evaluate PPO and explicit baselines on identical, held-out pipe seeds."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from flappyenv import EnvConfig, FlappyBirdEnv
from ppo import load_policy


def evaluate(network=None, config=None, episodes=30, seed=10000, baseline=None, deterministic=True, device="cpu"):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = config or EnvConfig()
    envs = [FlappyBirdEnv(config) for _ in range(episodes)]
    observations = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    generators = [np.random.default_rng(seed + i) for i in range(episodes)]
    active = list(range(episodes))
    returns = np.zeros(episodes)
    rows = [None] * episodes
    while active:
        if baseline is None:
            if network is None:
                raise ValueError("A policy or baseline is required")
            with torch.no_grad():
                states = torch.as_tensor(np.asarray([observations[i] for i in active]), device=device)
                actions = network.action_value(states, deterministic=deterministic)[0].cpu().numpy()
        else:
            actions = []
            for i in active:
                env = envs[i]
                if baseline == "random":
                    action = int(generators[i].random() < 0.5)
                elif baseline == "coast":
                    action = 0
                elif baseline == "heuristic":
                    p = next(p for p in env.pipes if p.x + config.pipe_width >= env.bird.x - config.bird_radius)
                    target = (p.top_height + p.bottom_y) / 2
                    action = int(env.bird.y > target + 25 and env.bird.velocity >= 0)
                else:
                    raise ValueError(f"Unknown baseline: {baseline}")
                actions.append(action)
        remaining = []
        for i, action in zip(active, actions):
            observations[i], reward, terminated, truncated, info = envs[i].step(int(action))
            returns[i] += reward
            if terminated or truncated:
                rows[i] = {"seed": seed+i, "score": info["score"], "return": float(returns[i]),
                           "length": info["frames"], "terminated": terminated, "truncated": truncated}
            else:
                remaining.append(i)
        active = remaining
    for env in envs:
        env.close()
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    return {"policy": baseline or ("ppo_greedy" if deterministic else "ppo_sampled"), "episodes": episodes,
            "seed_start": seed, "env_config": asdict(config), "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=1)) if episodes > 1 else 0.0,
            "median_score": float(np.median(scores)), "min_score": int(scores.min()), "max_score": int(scores.max()),
            "mean_return": float(returns.mean()), "mean_length": float(np.mean([r["length"] for r in rows])),
            "survival_rate": float(np.mean([r["truncated"] for r in rows])), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/ppo.pt")
    parser.add_argument("--baseline", choices=["random", "coast", "heuristic"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    network, metadata = (None, {}) if args.baseline else load_policy(args.model)
    settings = metadata.get("env_config", {})
    if args.max_steps is not None:
        settings = {**settings, "max_episode_steps": args.max_steps}
    result = evaluate(network, EnvConfig(**settings), args.episodes, args.seed, args.baseline, not args.stochastic)
    result["checkpoint"] = None if args.baseline else str(Path(args.model))
    result["training_steps"] = metadata.get("global_step")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "env_config")}, indent=2))


if __name__ == "__main__":
    main()
