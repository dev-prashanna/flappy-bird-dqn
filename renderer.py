"""Pygame presentation; all game rules live in flappyenv.py."""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame


class Renderer:
    def __init__(self, config, mode="human"):
        self.config, self.mode = config, mode
        pygame.font.init()
        if mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((config.width, config.height))
            pygame.display.set_caption("Flappy Bird | PPO Flight Lab")
        else:
            self.screen = pygame.Surface((config.width, config.height))
        self.title_font = pygame.font.Font(None, 42)
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()

    def text(self, value, position, color=(232, 242, 247), large=False):
        self.screen.blit((self.title_font if large else self.font).render(value, True, color), position)

    def draw(self, env, subtitle=None, overlay=None):
        c, screen = self.config, self.screen
        screen.fill((16, 29, 49))
        for y in range(80, c.height, 60):
            pygame.draw.line(screen, (23, 42, 62), (0, y), (c.width, y))
        for x in range(0, c.width, 60):
            pygame.draw.line(screen, (23, 42, 62), (x, 0), (x, c.height))
        for p in env.pipes:
            for rect in [pygame.Rect(p.x, 0, c.pipe_width, p.top_height),
                         pygame.Rect(p.x, p.bottom_y, c.pipe_width, c.height-p.bottom_y)]:
                pygame.draw.rect(screen, (43, 170, 137), rect, border_radius=5)
                pygame.draw.rect(screen, (91, 229, 175), rect, width=3, border_radius=5)
        b = env.bird
        center = (int(b.x), int(b.y))
        pygame.draw.circle(screen, (255, 195, 75), center, int(c.bird_radius))
        pygame.draw.circle(screen, (255, 229, 158), (int(b.x-4), int(b.y+4)), 7)
        pygame.draw.circle(screen, (255, 255, 255), (int(b.x+6), int(b.y-5)), 5)
        pygame.draw.circle(screen, (16, 29, 49), (int(b.x+8), int(b.y-5)), 2)
        pygame.draw.polygon(screen, (255, 130, 66), [(b.x+12, b.y), (b.x+23, b.y+4), (b.x+12, b.y+7)])
        panel = pygame.Surface((c.width, 78), pygame.SRCALPHA)
        panel.fill((10, 20, 35, 220))
        screen.blit(panel, (0, 0))
        self.text("FLIGHT LAB", (24, 12), large=True)
        self.text(subtitle or "Proximal Policy Optimization", (25, 49), (140, 174, 195))
        self.text(f"PIPES  {env.score:03d}", (c.width-180, 17), (91, 229, 175), large=True)
        if overlay:
            shade = pygame.Surface((c.width, c.height), pygame.SRCALPHA)
            shade.fill((9, 16, 28, 180))
            screen.blit(shade, (0, 0))
            for index, line in enumerate(overlay):
                font = self.title_font if index == 0 else self.font
                surface = font.render(line, True, (240, 245, 249))
                screen.blit(surface, (c.width//2-surface.get_width()//2, c.height//2-65+index*45))
        if self.mode == "human":
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

    def close(self):
        if self.mode == "human":
            pygame.display.quit()
