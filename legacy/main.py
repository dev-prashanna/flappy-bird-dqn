import random
import sys

import pygame
import torch

from double_dqn import DoubleDQN
from flappyenv import FlappyBirdEnv

WIDTH, HEIGHT = 800, 600
FPS = 60

FONT_COLOR = (255, 255, 255)
BG_COLOR = (135, 206, 235)


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
        return pygame.Rect(
            self.x - self.radius, self.y - self.radius,
            self.radius * 2, self.radius * 2
        )

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
        self.bottom_height = HEIGHT - self.bottom_y

    def move(self):
        self.x -= self.SPEED

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, self.PIPE_WIDTH, self.top_height)
        bottom_rect = pygame.Rect(self.x, self.bottom_y, self.PIPE_WIDTH, self.bottom_height)
        return top_rect, bottom_rect

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, self.PIPE_WIDTH, self.top_height))
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.bottom_y, self.PIPE_WIDTH, self.bottom_height))


def get_state(bird, pipes):
    next_pipe = None
    for pipe in pipes:
        if pipe.x + pipe.PIPE_WIDTH > bird.x:
            next_pipe = pipe
            break
    if next_pipe is None:
        next_pipe = pipes[0]
    return [
        bird.y / HEIGHT,
        bird.velocity / 10,
        (next_pipe.x - bird.x) / WIDTH,
        next_pipe.top_height / HEIGHT,
        next_pipe.bottom_y / HEIGHT,
    ]


def play_manual():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)
    bird = Bird(80, HEIGHT // 2)
    pipes = [Pipe(700)]
    score = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()
                if event.key == pygame.K_ESCAPE:
                    running = False

        bird.move()

        if bird.y + bird.radius > HEIGHT or bird.y - bird.radius < 0:
            running = False

        for pipe in pipes:
            pipe.move()

        if pipes[-1].x < 500:
            pipes.append(Pipe(900))

        if pipes[0].x < -60:
            pipes.pop(0)
            score += 1

        bird_rect = bird.get_rect()
        for pipe in pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                running = False

        screen.fill(BG_COLOR)
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)
        score_surface = font.render(str(score), True, FONT_COLOR)
        screen.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, 20))

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def play_ai(model_path="final_model.pth"):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)

    agent = DoubleDQN(state_dim=5, action_dim=2)
    agent.load(model_path)

    bird = Bird(80, HEIGHT // 2)
    pipes = [Pipe(700)]
    score = 0
    running = True

    print(f"AI mode - loaded model from {model_path}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        state = get_state(bird, pipes)
        action = agent.select_action(state, epsilon=0.0)
        if action == 1:
            bird.jump()

        bird.move()

        if bird.y + bird.radius > HEIGHT or bird.y - bird.radius < 0:
            running = False

        for pipe in pipes:
            pipe.move()

        if pipes[-1].x < 500:
            pipes.append(Pipe(900))

        if pipes[0].x < -60:
            pipes.pop(0)
            score += 1

        bird_rect = bird.get_rect()
        for pipe in pipes:
            top_rect, bottom_rect = pipe.get_rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                running = False

        screen.fill(BG_COLOR)
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)
        score_surface = font.render(str(score), True, FONT_COLOR)
        screen.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, 20))

        pygame.display.update()
        clock.tick(FPS)

    print(f"Game over. Score: {score}")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    if "--ai" in sys.argv:
        model = "best_model.pth"
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]
        play_ai(model)
    else:
        play_manual()
