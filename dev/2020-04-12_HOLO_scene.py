# -*- coding: utf-8 -*-

VERB = False
VERB = True

import sys
import numpy as np
import zmq

line_width = 3
RESOLUTION = 1000
z0 = .65 # in meter
s0 = .15 # normalized unit
VA_X = 30 * np.pi/180 # vertical visual angle (in radians) of the camera
VA_Y = 45 * np.pi/180 # horizontal visual angle (in radians) of the camera
screen_height, screen_width, viewing_distance  = .30, .45, z0
# https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
# fovy : Specifies the field of view angle, in degrees, in the y direction.
# on calcule
VA = 2. * np.arctan2(screen_height/2., viewing_distance) * 180. / np.pi
pc_min, pc_max = 0.001, 1000000.0
print(f'VA = {VA:.3f} deg')



#  Socket to talk to server
zmqcontext = zmq.Context()
print("Connecting to server…")
socket = zmqcontext.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def translate(message):
    x, y, s = message.split(', ')
    x, y, s = int(x), int(y), int(s) # str > int
    x, y, s = x/RESOLUTION, y/RESOLUTION, s/RESOLUTION
    x, y, s = x-.5, y-.5, s
    print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')

    z = z0 * s0 / s
    x = - z * np.tan(x * VA_X)
    y = - z * np.tan(y * VA_Y)
    print(f'x, y, z (Eye) = {x:.3f}, {y:.3f}, {z:.3f}')
    return x, y, z

import pyglet
display = pyglet.canvas.get_display()
print ("DEBUG: display client says display" , display)
screens = display.get_screens()
print ("DEBUG: display client says screens" , screens)
for i, screen in enumerate(screens):
    print('Screen %d: %dx%d at (%d,%d)' % (i, screen.width, screen.height, screen.x, screen.y))
N_screen = len(screens) # number of screens
assert N_screen == 1 # we should be running on one screen only


from pyglet.window import Window
fullscreen = False
fullscreen = True
window_0 = Window(screen=screens[0], fullscreen=fullscreen, resizable=True, vsync = True)
# window_0.set_exclusive_mouse()
import pyglet.gl as gl
from pyglet.gl.glu import gluLookAt

def on_resize(width, height):
    gl.glViewport(0, 0, width*2, height*2) # HACK for retina display ?
    gl.glEnable(gl.GL_BLEND)
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)#gl.GL_DONT_CARE)# gl.GL_NICEST)#
    # gl.glDisable(gl.GL_DEPTH_TEST)
    # gl.glDisable(gl.GL_LINE_SMOOTH)
    gl.glColor3f(1.0, 1.0, 1.0)

window_0.on_resize = on_resize
window_0.set_visible(True)
window_0.set_mouse_visible(False)


# scene geometry
gl.gluPerspective(VA, 1.0*window_0.width/window_0.height, pc_min, pc_max)
# gluLookAt(screen_height/2, screen_width/2, 0, screen_height/2, screen_width/2, viewing_distance, 0., 0, 1.0)
gl.glEnable(gl.GL_LINE_STIPPLE)

# opengl coordinates
# https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
N = 1000
particles = np.zeros((6, N), dtype='f') # x, y, z, x, y, z
# center
particles[0:3, :] += np.array([screen_width/2, screen_height/2, 0])[:, None]
particles[3:6, :] += np.array([screen_width/2, screen_height/2, 0])[:, None]
# particles[0:3, :] += np.random.randn(3, N) * screen_height / 4
# particles[3:6, :] += particles[0:3, :] + np.array([screen_height/8, 0, 0])[:, None]
# scatter
particles[0:2, :] += np.random.randn(2, N) * screen_height / 8

# random height
particles[3, :] = particles[0, :]
particles[4, :] = particles[1, :]
std_height = .23
particle_height = std_height*np.random.rand(N)
particles[5, :] = particles[2, :] + particle_height
N = particles.shape[1]
#

axis_particles = []
axis_particles.append([screen_width/3, screen_height/4, 0,
                       screen_width/3, 3*screen_height/4, 0])
axis_particles.append([2*screen_width/3, screen_height/4, 0,
                  2*screen_width/3, 3*screen_height/4, 0])
axis_particles = np.array(axis_particles).T

screen_particles = []
screen_particles.append([0, 0, 0,
                        screen_width, 0, 0])
screen_particles.append([0, screen_height, 0,
                        screen_width, screen_height, 0])
screen_particles.append([screen_width, screen_height, 0,
                        screen_width, 0, 0])
screen_particles.append([0, 0, 0,
                         0, screen_height, 0])
screen_particles = np.array(screen_particles).T

# https://github.com/Yuriy-Leonov/Pyglet_z_axis_issue/blob/master/main.py


from pyglet.graphics import draw

@window_0.event
def on_draw():
    global my_cx, my_cy, my_cz, particle_height, std_height, particles

    particle_momentum = .1
    particle_height *= 1 - particle_momentum
    particle_height += particle_momentum * std_height*np.random.rand(N)
    particles[5, :] = particles[2, :] + particle_height

    window_0.clear()

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    # https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
    # fovy : Specifies the field of view angle, in degrees, in the y direction.
    # aspect :Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is the ratio of x (width) to y (height).
    # zNear : Specifies the distance from the viewer to the near clipping plane (always positive).
    # zFar : Specifies the distance from the viewer to the far clipping plane (always positive).
    VA = 2. * np.arctan2(screen_height/2., my_cz) * 180. / np.pi

    gl.gluPerspective(VA, window_0.width/window_0.height, pc_min, pc_max)
    # gluLookAt(eyex,eyey,eyez,centx,centy,centz,upx,upy,upz)
    gluLookAt(my_cx, my_cy, my_cz,
              screen_width/2, screen_height/2, 0,
              0, 1, 0)

    gl.glLineWidth(line_width)

    # TODO : https://pyglet.readthedocs.io/en/latest/programming_guide/graphics.html#batched-rendering
    gl.glColor3f(1., 1., 1.)
    draw(2*N, gl.GL_LINES, ('v3f', particles.T.ravel().tolist()))

    gl.glColor3f(1., 0., 0.)
    draw(2*2, gl.GL_LINES, ('v3f', axis_particles.T.ravel().tolist()))

    gl.glColor3f(0., 1., 0.)
    draw(2*4, gl.GL_LINES, ('v3f', screen_particles.T.ravel().tolist()))

import time
tic = time.time()

def update(dt):
    global my_cx, my_cy, my_cz

    message = "ERROR"
    tic = time.time()
    #  Get the reply.
    while (message == "ERROR"):
        print("Sending request … GO!")
        socket.send(b"GO!")

        message = socket.recv()
        message = message.decode()
        print(message)

        if message == "ERROR":
            print(message)
            sys.exit()

    x, y, z = translate(message)

    toc = time.time()

    print(f'FPS:{1/(toc-tic):.1f}')

    my_cx, my_cy, my_cz = screen_width/2 + y, screen_height/2 + x, z
    # my_cx, my_cy, my_cz = screen_width/2 + viewing_distance*np.sin(2*np.pi*toc*.1), screen_height/2, viewing_distance*np.cos(2*np.pi*toc*.1)
    if VERB:
        print(f'x, y, z (Eye) = {my_cx:.3f}, {my_cy:.3f}, {my_cz:.3f}')
        print(f'DEBUG {pyglet.clock.get_fps():.3f}  fps')


pyglet.clock.schedule(update)
pyglet.app.run()
