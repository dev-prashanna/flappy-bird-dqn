"""Categorical actor--critic, time-limit-correct GAE, and clipped PPO."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 6
    minibatch_size: int = 256
    target_kl: float = 0.03
    hidden_size: int = 64


class ActorCritic(nn.Module):
    def __init__(self, observation_dim=8, hidden_size=64):
        super().__init__()
        def mlp(output):
            return nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, output))
        self.actor, self.critic = mlp(2), mlp(1)
        for net in (self.actor, self.critic):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        # Flapping every frame is fatal; this prior still samples both actions.
        with torch.no_grad():
            self.actor[-1].bias[1] = -2.0

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    def action_value(self, observations, actions=None, deterministic=False):
        distribution = Categorical(logits=self.actor(observations))
        if actions is None:
            actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), self.value(observations)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):
    """Arrays are [time, environment]. next_values uses *pre-reset* observations.

    A true death suppresses bootstrapping. Either episode boundary stops the
    recursive trace, preventing the next episode's rewards leaking backward.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * (1 - np.logical_or(terminated[t], truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


class PPO:
    def __init__(self, config=None, device="cpu"):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.network = ActorCritic(hidden_size=self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate, eps=1e-5)

    def update(self, batch):
        c = self.config
        data = {k: torch.as_tensor(v, device=self.device).flatten(0, 1) for k, v in batch.items()}
        n = len(data["actions"])
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        stats = []
        stop = False
        for _ in range(c.update_epochs):
            for indices in torch.randperm(n, device=self.device).split(c.minibatch_size):
                _, log_prob, entropy, value = self.network.action_value(data["observations"][indices], data["actions"][indices])
                log_ratio = log_prob - data["log_probs"][indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                if c.target_kl > 0 and kl > c.target_kl:
                    stop = True
                    break
                adv = advantages[indices]
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1-c.clip_coef, 1+c.clip_coef) * adv).mean()
                value_loss = 0.5 * (value - data["returns"][indices]).square().mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss is not finite")
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), c.max_grad_norm)
                self.optimizer.step()
                stats.append([policy_loss.item(), value_loss.item(), entropy.mean().item(), kl.item(),
                              ((ratio - 1).abs() > c.clip_coef).float().mean().item(), float(grad_norm)])
            if stop:
                break
        names = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"]
        metrics = dict(zip(names, np.mean(stats, axis=0).tolist())) if stats else dict.fromkeys(names, 0.0)
        metrics["gradient_steps"] = len(stats)
        metrics["kl_stopped"] = stop
        return metrics

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format_version": 1, "algorithm": "PPO", "model": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(), **metadata}, temporary)
        temporary.replace(path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "PPO" or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a version-1 PPO checkpoint; legacy DQN weights are incompatible")
    network = ActorCritic(hidden_size=checkpoint["ppo_config"]["hidden_size"]).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network, checkpoint
