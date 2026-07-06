import random

import pygame 
import sys 

width, height =800, 600
fps=60

class Bird:

    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.velocity=0
        self.radius=15
        self.color=(255,255,0)
        self.gravity=0.5
        self.jump_strength=-8

    def jump(self):
        self.velocity=self.jump_strength 

    def move(self):
        self.velocity+=self.gravity 
        self.y+=self.velocity 

    def get_rect(self):
        return pygame.Rect(self.x-self.radius,self.y-self.radius,self.radius*2,self.radius*2)
    
    
    def draw(self,screen):
        pygame.draw.circle(screen,self.color,(self.x,self.y),self.radius)

class Pipe:
    PIPE_WIDTH=60
    GAP=150
    SPEED=3

    def __init__(self,x):
        self.x=x
        self.top_height=random.randint(100,400)
        self.bottom_y=self.top_height+self.GAP
        self.bottom_height=height-self.bottom_y

    def move(self):
        self.x-=self.SPEED


    def draw(self,screen):
        pygame.draw.rect(screen,(0,255,0),(self.x,0,self.PIPE_WIDTH,self.top_height))
        pygame.draw.rect(screen,(0,255,0),(self.x,self.bottom_y,self.PIPE_WIDTH,self.bottom_height))
    


def main():
    pygame.init()
    screen=pygame.display.set_mode((width, height))
    clock=pygame.time.Clock()
    bird=Bird(80,height//2)
    pipes=[Pipe(700)]
    running=True 

    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False 
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_SPACE:
                    bird.jump()
        
        bird.move()
        
        for pipe in pipes:
         pipe.move()
         if pipes[-1].x<500:
             pipes.append(Pipe(900))

         if pipes[0].x<-60:
                pipes.pop(0)

        bird_rect=pygame.Rect(bird.x-bird.radius,bird.y-bird.radius,bird.radius*2,bird.radius*2)
        screen.fill((135,206,235))
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    sys.exit()



    
if __name__=="__main__":
    main()        