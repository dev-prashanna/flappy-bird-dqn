"""Empirical confirmation of the in-gap clearance band (paper, Proposition 6)."""
from flappyenv import EnvConfig, FlappyBirdEnv

THRESHOLD = 25.0
HALF_ARC = 60.0
FALL_CAP = 10.0


def heuristic_action(env):
    c = env.config
    pipe = next(p for p in env.pipes if p.x + c.pipe_width >= env.bird.x - c.bird_radius)
    target = (pipe.top_height + pipe.bottom_y) / 2
    return int(env.bird.y > target + THRESHOLD and env.bird.velocity >= 0), pipe


def test_in_gap_band_holds_after_entry():
    config = EnvConfig()
    overlap_frames = in_band_frames = entry_outside = 0
    episodes_with_violation = 0
    for seed in range(10000, 10030):
        env = FlappyBirdEnv(config)
        env.reset(seed=seed)
        pipe_id, inside, violated = None, False, False
        while True:
            action, pipe = heuristic_action(env)
            _, _, terminated, truncated, _ = env.step(action)
            overlap = (env.bird.x + config.bird_radius >= pipe.x
                       and env.bird.x - config.bird_radius <= pipe.x + config.pipe_width)
            if id(pipe) != pipe_id:
                pipe_id, inside = id(pipe), False
            if overlap and not terminated:
                overlap_frames += 1
                offset = env.bird.y - (pipe.top_height + pipe.bottom_y) / 2
                in_band = -HALF_ARC + THRESHOLD < offset <= FALL_CAP + THRESHOLD
                if in_band:
                    inside, in_band_frames = True, in_band_frames + 1
                elif inside:
                    violated = True
                else:
                    entry_outside += 1
            if terminated or truncated:
                episodes_with_violation += int(violated)
                break
        env.close()
    assert overlap_frames > 50000
    assert episodes_with_violation == 0
    assert in_band_frames / overlap_frames > 0.999
    assert entry_outside < 100


def test_flap_arcs_and_velocity_bounds():
    env = FlappyBirdEnv(EnvConfig())
    env.reset(seed=0)
    velocities, rise = [], 0.0
    for t in range(40):
        _, _, terminated, truncated, _ = env.step(1 if t == 0 else 0)
        velocities.append(env.bird.velocity)
        rise = min(rise, env.bird.y - 300.0)
        if terminated or truncated:
            break
    assert velocities[0] == -7.5
    assert all(v >= -7.5 for v in velocities)
    assert rise == -60.0
    assert env.bird.y - 300.0 > -60.0
    env.close()
