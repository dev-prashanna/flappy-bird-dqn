"""Train on-policy PPO without rendering; run `python train_net.py --help`."""
import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evaluate import evaluate
from flappyenv import EnvConfig, FlappyBirdEnv
from ppo import PPO, PPOConfig, compute_gae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=25, help="PPO updates between validation runs")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="Continue weights/optimizer, with fresh seeded environments and RNGs")
    args = parser.parse_args()
    if min(args.total_timesteps, args.num_envs, args.num_steps, args.epochs, args.minibatch_size,
           args.eval_every, args.eval_episodes, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Training sizes, intervals, threads, and learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        parser.error("Output already contains a run; choose a new --output directory (also for --resume)")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    c = PPOConfig(learning_rate=args.learning_rate, update_epochs=args.epochs, minibatch_size=args.minibatch_size)
    ec = EnvConfig(max_episode_steps=args.max_episode_steps)
    agent = PPO(c, args.device)
    global_step = 0
    if args.resume:
        from ppo import load_policy
        network, checkpoint = load_policy(args.resume, args.device)
        if checkpoint["ppo_config"] != asdict(c) or checkpoint["env_config"] != asdict(ec):
            parser.error("Resume requires matching PPO and environment settings")
        agent.network.load_state_dict(network.state_dict())
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
    config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "ppo_config": asdict(c), "env_config": asdict(ec), "torch_version": str(torch.__version__),
              "numpy_version": np.__version__, "starting_step": global_step}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    envs = [FlappyBirdEnv(ec) for _ in range(args.num_envs)]
    observations = np.stack([e.reset(seed=args.seed+i)[0] for i, e in enumerate(envs)])
    episode_returns = np.zeros(args.num_envs)
    recent_scores, recent_returns = [], []
    start = time.monotonic()
    best = (-math.inf, -math.inf)
    updates = math.ceil(args.total_timesteps / (args.num_envs * args.num_steps))
    fields = ["update", "global_step", "mean_score_100", "mean_return_100", "episodes", "sps",
              "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
              "gradient_steps", "kl_stopped", "validation_score", "validation_return"]
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for update in range(1, updates+1):
            fraction = 1 - (update-1) / updates
            agent.optimizer.param_groups[0]["lr"] = c.learning_rate * fraction
            shape = (args.num_steps, args.num_envs)
            observations_buf = np.empty((*shape, 8), dtype=np.float32)
            next_observations = np.empty_like(observations_buf)
            actions_buf = np.empty(shape, dtype=np.int64)
            log_probs, values, rewards = [np.empty(shape, dtype=np.float32) for _ in range(3)]
            terminated, truncated = [np.zeros(shape, dtype=bool) for _ in range(2)]
            for t in range(args.num_steps):
                observations_buf[t] = observations
                with torch.no_grad():
                    actions, lp, _, v = agent.network.action_value(torch.as_tensor(observations, device=agent.device))
                actions_buf[t], log_probs[t], values[t] = actions.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
                for i, env in enumerate(envs):
                    obs, reward, term, trunc, info = env.step(int(actions_buf[t, i]))
                    next_observations[t, i] = obs
                    rewards[t, i], terminated[t, i], truncated[t, i] = reward, term, trunc
                    episode_returns[i] += reward
                    if term or trunc:
                        recent_scores.append(info["score"])
                        recent_returns.append(float(episode_returns[i]))
                        episode_returns[i] = 0
                        obs, _ = env.reset()
                    observations[i] = obs
                global_step += args.num_envs
            with torch.no_grad():
                next_values = agent.network.value(torch.as_tensor(next_observations.reshape(-1, 8), device=agent.device)).cpu().numpy().reshape(shape)
            advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, c.gamma, c.gae_lambda)
            metrics = agent.update({"observations": observations_buf, "actions": actions_buf,
                                    "log_probs": log_probs, "advantages": advantages, "returns": returns})
            metadata = {"ppo_config": asdict(c), "env_config": asdict(ec), "global_step": global_step, "seed": args.seed}
            validation = None
            if update % args.eval_every == 0 or update == updates:
                validation = evaluate(agent.network, ec, args.eval_episodes, args.eval_seed, device=agent.device)
                quality = (validation["mean_score"], validation["mean_return"])
                if quality > best:
                    best = quality
                    agent.save(args.output / "best.pt", {**metadata, "validation": validation})
                (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
            row = {"update": update, "global_step": global_step,
                   "mean_score_100": np.mean(recent_scores[-100:]) if recent_scores else 0,
                   "mean_return_100": np.mean(recent_returns[-100:]) if recent_returns else 0,
                   "episodes": len(recent_scores), "sps": int((global_step-config["starting_step"])/(time.monotonic()-start)),
                   **metrics, "validation_score": validation["mean_score"] if validation else "",
                   "validation_return": validation["mean_return"] if validation else ""}
            writer.writerow(row)
            file.flush()
            agent.save(args.output / "latest.pt", metadata)
            if update == 1 or update % 10 == 0 or validation:
                print(f"update={update}/{updates} steps={global_step} score100={row['mean_score_100']:.2f} "
                      f"return100={row['mean_return_100']:.1f} sps={row['sps']} "
                      f"validation={row['validation_score']}", flush=True)
        agent.save(args.output / "final.pt", metadata)
    for env in envs:
        env.close()
    print(f"Saved best validation policy and final policy to {args.output}")


if __name__ == "__main__":
    main()
