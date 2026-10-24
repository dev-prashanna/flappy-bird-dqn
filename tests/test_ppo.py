from dataclasses import asdict

import numpy as np
import torch

from flappyenv import EnvConfig
from ppo import PPO, PPOConfig, compute_gae, load_policy

torch.set_num_threads(1)


def test_gae_termination_truncation_and_no_cross_episode_leakage():
    rewards = np.array([[1, 1], [100, 100]], np.float32)
    values = np.array([[2, 2], [0, 0]], np.float32)
    next_values = np.array([[10, 10], [0, 0]], np.float32)
    terminated = np.array([[True, False], [True, True]])
    truncated = np.array([[False, True], [False, False]])
    advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.9)
    np.testing.assert_allclose(advantages[0], [-1, 8])
    np.testing.assert_allclose(returns[0], [1, 10])


def test_gae_matches_hand_calculated_trace():
    rewards = np.array([[1], [2]], np.float32)
    zeros = np.zeros_like(rewards)
    advantages, returns = compute_gae(rewards, zeros, zeros, zeros, zeros, gamma=0.9, gae_lambda=0.8)
    np.testing.assert_allclose(advantages[:, 0], [2.44, 2])
    np.testing.assert_array_equal(advantages, returns)


def test_policy_update_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    agent = PPO(PPOConfig(update_epochs=2, minibatch_size=8))
    obs = torch.randn(4, 4, 8)
    with torch.no_grad():
        actions, lp, _, _ = agent.network.action_value(obs)
    before = {k: v.clone() for k, v in agent.network.state_dict().items()}
    stats = agent.update({"observations": obs.numpy(), "actions": actions.numpy(), "log_probs": lp.numpy(),
                          "advantages": np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4),
                          "returns": np.ones((4, 4), np.float32)})
    assert stats["gradient_steps"] > 0
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(v, before[k]) for k, v in agent.network.state_dict().items())
    path = tmp_path / "model.pt"
    agent.save(path, {"ppo_config": asdict(agent.config), "env_config": asdict(EnvConfig()), "global_step": 16})
    loaded, checkpoint = load_policy(path)
    assert checkpoint["global_step"] == 16
    torch.testing.assert_close(loaded.actor(obs), agent.network.actor(obs))
    torch.testing.assert_close(loaded.value(obs), agent.network.value(obs))


def test_constant_advantages_do_not_produce_nan():
    agent = PPO(PPOConfig(update_epochs=1, minibatch_size=4))
    obs = np.zeros((2, 2, 8), np.float32)
    with torch.no_grad():
        actions, lp, _, _ = agent.network.action_value(torch.from_numpy(obs))
    stats = agent.update({"observations": obs, "actions": actions.numpy(), "log_probs": lp.numpy(),
                          "advantages": np.zeros((2, 2), np.float32), "returns": np.zeros((2, 2), np.float32)})
    assert np.isfinite(stats["policy_loss"])
