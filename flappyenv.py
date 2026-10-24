"""Shared deterministic physics for headless learning and interactive play."""
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WIDTH, HEIGHT = 800, 600


@dataclass(frozen=True)
class EnvConfig:
    width: int = WIDTH
    height: int = HEIGHT
    gravity: float = 0.5
    flap_velocity: float = -8.0
    max_fall_speed: float = 10.0
    bird_x: float = 80.0
    bird_radius: float = 15.0
    pipe_width: float = 60.0
    pipe_gap: float = 150.0
    pipe_speed: float = 3.0
    pipe_spacing: float = 260.0
    first_pipe_x: float = 350.0
    min_pipe_top: int = 80
    max_pipe_top: int = 370
    survival_reward: float = 0.1
    pass_reward: float = 5.0
    death_reward: float = -1.0
    max_episode_steps: int = 6000

    def __post_init__(self):
        if not (self.width > self.bird_x + self.bird_radius > 2 * self.bird_radius > 0):
            raise ValueError("Bird must fit horizontally inside the screen")
        if not (self.bird_radius < self.min_pipe_top <= self.max_pipe_top < self.height - self.pipe_gap - self.bird_radius):
            raise ValueError("Pipe gaps must fit inside the screen")
        if self.pipe_gap <= 2 * self.bird_radius or self.pipe_width <= 0:
            raise ValueError("Pipe gap must fit the bird; pipe width must be positive")
        if self.pipe_speed <= 0 or self.pipe_spacing <= self.pipe_width + 2 * self.bird_radius:
            raise ValueError("Pipe speed and spacing must allow separate obstacles")
        if self.first_pipe_x <= self.bird_x + self.bird_radius or self.max_episode_steps < 1:
            raise ValueError("Invalid initial pipe position or episode length")
        if self.gravity <= 0 or self.flap_velocity >= 0 or self.max_fall_speed <= 0:
            raise ValueError("Invalid physics constants")


@dataclass
class Bird:
    x: float
    y: float
    velocity: float = 0.0


@dataclass
class Pipe:
    x: float
    top_height: float
    bottom_y: float
    passed: bool = False


class FlappyBirdEnv(gym.Env):
    """One action advances one 1/60-second physics frame; 0=coast, 1=flap.

    Collision uses the bird's axis-aligned bounding square, consistently in all
    modes. Observations expose both approaching pipes, including the trailing
    edge of an overlapping pipe, so switching targets cannot hide a collision.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or EnvConfig()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unknown render mode: {render_mode}")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self._renderer = None
        self._done = True

    def _new_pipe(self, x):
        c = self.config
        top = int(self.np_random.integers(c.min_pipe_top, c.max_pipe_top + 1))
        return Pipe(x, top, top + c.pipe_gap)

    def _get_state(self):
        c = self.config
        upcoming = [p for p in self.pipes if p.x + c.pipe_width >= self.bird.x - c.bird_radius]
        values = [self.bird.y / c.height, self.bird.velocity / 10.0]
        for p in upcoming[:2]:
            values.extend([(p.x - self.bird.x) / c.width, p.top_height / c.height, p.bottom_y / c.height])
        return np.asarray(values, dtype=np.float32)

    def _info(self):
        return {"score": self.score, "frames": self.frame_count}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        self.bird = Bird(c.bird_x, c.height / 2)
        self.pipes = [self._new_pipe(c.first_pipe_x + i * c.pipe_spacing) for i in range(3)]
        self.score = self.frame_count = 0
        self._done = False
        return self._get_state(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        c, b = self.config, self.bird
        self.frame_count += 1
        if action == 1:
            b.velocity = c.flap_velocity
        b.velocity = min(b.velocity + c.gravity, c.max_fall_speed)
        b.y += b.velocity
        for p in self.pipes:
            p.x -= c.pipe_speed
        terminated = b.y - c.bird_radius <= 0 or b.y + c.bird_radius >= c.height
        for p in self.pipes:
            overlap = b.x + c.bird_radius >= p.x and b.x - c.bird_radius <= p.x + c.pipe_width
            if overlap and (b.y - c.bird_radius <= p.top_height or b.y + c.bird_radius >= p.bottom_y):
                terminated = True
        passed = 0
        if not terminated:
            for p in self.pipes:
                if not p.passed and p.x + c.pipe_width < b.x - c.bird_radius:
                    p.passed = True
                    self.score += 1
                    passed += 1
        self.pipes = [p for p in self.pipes if p.x + c.pipe_width >= 0]
        while len([p for p in self.pipes if p.x + c.pipe_width >= b.x - c.bird_radius]) < 3:
            self.pipes.append(self._new_pipe(self.pipes[-1].x + c.pipe_spacing))
        reward = c.death_reward if terminated else c.survival_reward + c.pass_reward * passed
        truncated = self.frame_count >= c.max_episode_steps and not terminated
        self._done = terminated or truncated
        return self._get_state(), float(reward), bool(terminated), bool(truncated), self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from renderer import Renderer
            self._renderer = Renderer(self.config, self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def config_dict(self):
        return asdict(self.config)
