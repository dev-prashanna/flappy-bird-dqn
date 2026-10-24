import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flappyenv import EnvConfig, FlappyBirdEnv, Pipe


def test_gymnasium_contract():
    check_env(FlappyBirdEnv(), skip_render_check=True)


def test_seeded_trajectory_and_local_rng():
    a, b = FlappyBirdEnv(), FlappyBirdEnv()
    np.testing.assert_array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
    FlappyBirdEnv().reset(seed=99)  # must not disturb either environment's RNG
    for t in range(250):
        action = int(t % 25 == 0)
        sa, ra, ta, ca, ia = a.step(action)
        sb, rb, tb, cb, ib = b.step(action)
        np.testing.assert_array_equal(sa, sb)
        assert (ra, ta, ca, ia) == (rb, tb, cb, ib)
        if ta or ca:
            a.reset()
            b.reset()


def test_gravity_flap_and_floor_termination():
    env = FlappyBirdEnv()
    env.reset(seed=1)
    env.step(1)
    assert env.bird.velocity == -7.5
    assert env.bird.y == 292.5
    env.bird.y, env.bird.velocity = 585, 0
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated and reward == -1
    with pytest.raises(RuntimeError):
        env.step(0)


def test_time_limit_is_not_death():
    env = FlappyBirdEnv(EnvConfig(max_episode_steps=1))
    env.reset(seed=1)
    _, reward, terminated, truncated, _ = env.step(0)
    assert not terminated and truncated and reward == 0.1


def test_score_once_only_after_clearing_and_no_hidden_trailing_collision():
    env = FlappyBirdEnv()
    env.reset(seed=1)
    env.pipes[0] = Pipe(7, 200, 350)
    _, reward, terminated, _, info = env.step(0)
    assert not terminated and info["score"] == 1 and reward == 5.1
    assert env.step(0)[4]["score"] == 1
    env.reset(seed=1)
    env.pipes[0] = Pipe(18, 310, 460)  # trailing edge behind center, still touches bird
    assert env._get_state()[2] < 0
    assert env.step(0)[2]
    assert env.score == 0


def test_observations_remain_valid_across_many_pipe_switches():
    env = FlappyBirdEnv()
    env.reset(seed=100)
    for _ in range(2000):
        pipe = next(p for p in env.pipes if p.x + 60 >= env.bird.x - 15)
        env.bird.y = (pipe.top_height + pipe.bottom_y) / 2
        env.bird.velocity = 0
        observation, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        assert observation.shape == (8,)
        assert env.observation_space.contains(observation)
    assert env.score > 10


def test_render_array():
    env = FlappyBirdEnv(render_mode="rgb_array")
    env.reset(seed=1)
    image = env.render()
    assert image.shape == (600, 800, 3) and image.dtype == np.uint8
    assert image.std() > 0
    env.close()


def test_bad_action_and_config():
    env = FlappyBirdEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(2)
    with pytest.raises(ValueError):
        EnvConfig(pipe_gap=10)
