import random
import numpy as np


WIDTH, HEIGHT = 800, 600


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.radius = 15
        self.gravity = 0.5
        self.jump_strength = -8

    def jump(self):
        self.velocity = self.jump_strength

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def collides_with(self, top_h, bot_y, pipe_x):
        if self.y + self.radius > HEIGHT or self.y - self.radius < 0:
            return True
        if self.x + self.radius > pipe_x and self.x - self.radius < pipe_x + Pipe.PIPE_WIDTH:
            if self.y - self.radius < top_h or self.y + self.radius > bot_y:
                return True
        return False


class Pipe:
    PIPE_WIDTH = 60
    GAP = 150
    SPEED = 3

    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(100, 400)
        self.bottom_y = self.top_height + self.GAP
        self.bottom_height = HEIGHT - self.bottom_y

    def move(self):
        self.x -= self.SPEED


class FlappyBirdEnv:
    def __init__(self):
        self.state_shape = (5,)
        self.action_space = [0, 1]
        self.reset()

    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + pipe.PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[0]

    def _get_state(self):
        next_pipe = self._get_next_pipe()
        return np.array([
            self.bird.y / HEIGHT,
            self.bird.velocity / 10,
            (next_pipe.x - self.bird.x) / WIDTH,
            next_pipe.top_height / HEIGHT,
            next_pipe.bottom_y / HEIGHT
        ], dtype=np.float32)

    def reset(self):
        self.bird = Bird(80, HEIGHT // 2)
        self.pipes = [Pipe(350)]
        self.score = 0
        self.frame_count = 0
        return self._get_state()

    def step(self, action):
        self.frame_count += 1

        if action == 1:
            self.bird.jump()

        self.bird.move()

        for pipe in self.pipes:
            pipe.move()

        if self.pipes[-1].x < 300:
            self.pipes.append(Pipe(WIDTH))

        done = False
        reward = 1.0

        if self.pipes[0].x < -Pipe.PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1
            reward += 10.0

        if self.bird.y + self.bird.radius > HEIGHT or self.bird.y - self.bird.radius < 0:
            done = True
            reward = -5.0

        for pipe in self.pipes:
            if self.bird.collides_with(pipe.top_height, pipe.bottom_y, pipe.x):
                done = True
                reward = -5.0
                break

        return self._get_state(), reward, done, {}
