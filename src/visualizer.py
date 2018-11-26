# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *
import cv2
import numpy as np
from threading import Thread
viewport=400,400
def display(obj):
    srf = pygame.display.set_mode(viewport, OPENGLBLIT | DOUBLEBUF)
    # srf.blit(frame, (0,0))
    # pygame.display.update()
    # srf.fill([255,255,255])
    glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

    print("printing object")

    # LOAD OBJECT AFTER PYGAME INIT
    # camera = cv2.VideoCapture(0)
    obj = OBJ(obj, swapyz=False)

    # ret, frame = camera.read()

    pygame.init()
    clock = pygame.time.Clock()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(90, width/float(height), 1, 200.0)
    # glScale(0.2, 0.2, 0.0);
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)


    rx, ry = (0,0)
    tx, ty = (2,2)
    zpos = 30
    rotate = move = False
    while 1:
        clock.tick(0)
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                sys.exit()
            elif e.type == MOUSEBUTTONDOWN:
                if e.button == 4: zpos = max(1, zpos-1)
                elif e.button == 5: zpos += 1
                elif e.button == 1: rotate = True
                elif e.button == 3: move = True
            elif e.type == MOUSEBUTTONUP:
                if e.button == 1: rotate = False
                elif e.button == 3: move = False
            elif e.type == MOUSEMOTION:
                i, j = e.rel
                if rotate:
                    rx += i
                    ry += j
                if move:
                    tx += i
                    ty -= j

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # RENDER OBJECT
        # print(tx,ty)
        glTranslate(tx, ty, - zpos)
        glRotate(ry, 1, 0, 0)
        glRotate(rx, 0, 1, 0)
        # print(obj.gl_list)
        glCallList(obj.gl_list)

        pygame.display.flip()


# frame = pygame.surfarray.make_surface(frame)\
# sys.path.append('../model/model1')
# display("../model/model1/OBJ.obj")

if __name__ == '__main__':
    display(sys.argv[1])
