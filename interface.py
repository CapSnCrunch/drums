import pygame
import numpy as np
from drums import *

win = pygame.display.set_mode((500, 500))
pygame.display.set_caption('Drum Eigenvalue Calculator')
pygame.font.init()

class Vertex:
    def __init__(self, x, y, color = (0,0,0)):
        self.pos = np.array([x, y])
        self.color = color
    
    def draw(self):
        pygame.draw.circle(win, self.color, self.pos, 15)

if __name__ == '__main__':
    
    vertices = []

    run = True
    drag = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drag = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drag = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    mouseX, mouseY = np.array(pygame.mouse.get_pos())
                    vertices.append(Vertex(mouseX, mouseY))
                if event.key == pygame.K_SLASH:
                    x, y = [], []
                    for v in vertices:
                        x.append(v.pos[0])
                        y.append(v.pos[1])
                    x.append(vertices[-1].pos[0])
                    y.append(vertices[-1].pos[1])
                    x, y = np.array(x), np.array(y)
                    scale = max([np.max(x), np.max(y)])
                    x, y = 5 * x / scale, 5 * y / scale

                    s = Solver(x, y, ngrid = 16)
                    print(s.solve())

        if drag:
            mouse = np.array(pygame.mouse.get_pos())
            for v in vertices:
                if np.linalg.norm(v.pos - mouse) < 15:
                    intersecting = False
                    for v2 in vertices:
                        if v != v2 and np.linalg.norm(v2.pos - mouse) < 28:
                            intersecting = True
                    if not intersecting:
                        v.pos = mouse

        win.fill((255, 255, 255))
        for i in range(len(vertices)):
            vertices[i].draw()
            pygame.draw.line(win, (255*(i+1==len(vertices)),0,0), vertices[i].pos, vertices[(i+1)%len(vertices)].pos, 2)

        pygame.display.update()

        