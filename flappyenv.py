import random
import pygame
import numpy as np

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.radius = 15
        self.color = (255, 255, 0)
        self.gravity = 0.5
        self.jump_strength = -8

    def jump(self):
        self.velocity = self.jump_strength

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Pipe:
    PIPE_WIDTH = 60
    GAP = 150
    SPEED = 3

    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(100, 400)
        self.bottom_y = self.top_height + self.GAP
        self.bottom_height = 600 - self.bottom_y

    def move(self):
        self.x -= self.SPEED

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, self.PIPE_WIDTH, self.top_height)
        bottom_rect = pygame.Rect(self.x, self.bottom_y, self.PIPE_WIDTH, self.bottom_height)
        return top_rect, bottom_rect

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, self.PIPE_WIDTH, self.top_height))
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.bottom_y, self.PIPE_WIDTH, self.bottom_height))

WIDTH, HEIGHT = 800, 600
FPS=60

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen=pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock=pygame.time.Clock()


        self.state_shape=(5,)
        self.action_space=[0,1]
        self.reset()

    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + pipe.PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[0]


    def _get_state(self):

        next_pipe=self._get_next_pipe()
        state=np.array([
            self.bird.y /HEIGHT,
            self.bird.velocity /10,
            (next_pipe.x - self.bird.x) / WIDTH,
            next_pipe.top_height / HEIGHT,
            next_pipe.bottom_y / HEIGHT
        ], dtype=np.float32)

        return state

    def reset(self):

        self.bird=Bird(80, HEIGHT//2)
        self.pipes=[Pipe(700)]
        self.score=0
        self.frame_count=0

        return self._get_state()

    def step(self, action):
        self.frame_count+=1

        if action==1:
            self.bird.jump()

        self.bird.move()

        for pipe in self.pipes:
            pipe.move()

        if self.pipes[-1].x<500:
            self.pipes.append(Pipe(WIDTH))

        if self.pipes[0].x < -Pipe.PIPE_WIDTH:
            self.pipes.pop(0)
            self.score+=1

        done=False
        reward=0.1

        if self.bird.y + self.bird.radius > HEIGHT or self.bird.y -self.bird.radius < 0:
            done=True 
            reward =-10

        bird_rect = self.bird.get_rect()
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                done = True
                reward = -100.0
                break

        next_state=self._get_state()
        return next_state, reward, done, {}

    