line_width = 3
VERB = False
import pyglet
# platform = pyglet.window.get_platform()
# print ("DEBUG: display client says platform" , platform)
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
win_0 = Window(screen=screens[0], fullscreen=fullscreen, resizable=True, vsync = True)
# win_0.set_exclusive_mouse()
import pyglet.gl as gl
from pyglet.gl.glu import gluLookAt

def on_resize(width, height):
    gl.glViewport(0, 0, width*2, height*2) # HACK for retina display ?
    gl.glEnable(gl.GL_BLEND)
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)#gl.GL_DONT_CARE)# gl.GL_NICEST)#
    # gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_LINE_SMOOTH)
    gl.glColor3f(1.0, 1.0, 1.0)

win_0.on_resize = on_resize
win_0.set_visible(True)
# win_0.set_mouse_visible(False)
# gl.glMatrixMode(gl.GL_MODELVIEW)
# gl.glLoadIdentity()


# scene geometry
import numpy as np
screen_height, screen_width, viewing_distance  = .30, .45, .60
# on calcule
VA = 2. * np.arctan2(screen_width/2., viewing_distance) * 180. / np.pi
pc_min, pc_max = 0.001, 1000000.0
# gl.gluPerspective(VA, 1.0*win_0.width/win_0.height, pc_min, pc_max)
# gluLookAt(screen_height/2, screen_width/2, 0, screen_height/2, screen_width/2, viewing_distance, 0., 0, 1.0)
#gl.glEnable(gl.GL_LINE_STIPPLE)
print(f'VA = {VA:.3f} deg')


# opengl coordinates
# https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
N = 1000
particles = np.zeros((6, N), dtype='f') # x, y, z, x, y, z
particles[0:3, :] += np.array([screen_width/2, screen_height/2, 0])[:, None]
particles[3:6, :] += np.array([screen_width/2, screen_height/2, 0])[:, None]
# particles[0:3, :] += np.random.randn(3, N) * screen_height / 4
# particles[3:6, :] += particles[0:3, :] + np.array([screen_height/8, 0, 0])[:, None]
particles[0:2, :] += np.random.randn(2, N) * screen_height / 8
# particles[3:5, :] += particles[0:2, :] + np.array([screen_height/8, 0])[:, None]
particles[3, :] = particles[0, :]
particles[4, :] = particles[1, :]
particles[5, :] = particles[2, :] + .23*np.random.rand(N)
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

@win_0.event
def on_draw():
    global my_cx, my_cy, my_cz
    win_0.clear()

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    # https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
    # fovy : Specifies the field of view angle, in degrees, in the y direction.
    # aspect :Specifies the aspect ratio that determines the field of view in the x direction. The aspect ratio is the ratio of x (width) to y (height).
    # zNear : Specifies the distance from the viewer to the near clipping plane (always positive).
    # zFar : Specifies the distance from the viewer to the far clipping plane (always positive).
    gl.gluPerspective(VA, win_0.width/win_0.height, pc_min, pc_max)
    # gluLookAt(eyex,eyey,eyez,centx,centy,centz,upx,upy,upz)
    gluLookAt(my_cx, my_cy, my_cz,
              screen_width/2, screen_height/2, 0,
              0, 1, 0)

    gl.glLineWidth(line_width)

    gl.glColor3f(1., 1., 1.)
    draw(2*N, gl.GL_LINES, ('v3f', particles.T.ravel().tolist()))

    gl.glColor3f(1., 0., 0.)
    draw(2*2, gl.GL_LINES, ('v3f', axis_particles.T.ravel().tolist()))

    gl.glColor3f(0., 1., 0.)
    draw(2*4, gl.GL_LINES, ('v3f', screen_particles.T.ravel().tolist()))

import time
tic = time.time()

def callback(dt):
    global my_cx, my_cy, my_cz
    toc = time.time() - tic
    # modul = 1 + .3 * np.sin(2*np.pi*toc*.1)
    # my_cx, my_cy, my_cz = screen_width/2, 2*screen_height/2, viewing_distance*modul
    modul = 1 + .3 * np.sin(2*np.pi*toc*.1)
    my_cx, my_cy, my_cz = screen_width/2 + viewing_distance*np.sin(2*np.pi*toc*.1), screen_height/2, viewing_distance*np.cos(2*np.pi*toc*.1)
    if VERB: print(f'DEBUG {pyglet.clock.get_fps():.3f}  fps')

pyglet.clock.schedule(callback)
pyglet.app.run()
