#! /usr/bin/python

"""

OpenLoop.py: detecting a salient area and moving toward it

Laurent Perrinet, 2010. Credits: see http://www.incm.cnrs-mrs.fr/LaurentPerrinet/SimpleCellDemo

$Id$

"""

import motmot.cam_iface.cam_iface_ctypes as cam_iface
from motmot.imops.imops import yuv422_to_mono8 as convert

import numpy as np
import time

from pyglet.gl import *
from pyglet import window
from pyglet import image
from pyglet import graphics
from pygarrayimage.arrayimage import ArrayInterfaceImage
import scipy.ndimage


from SimpleReceptiveField import calibrate, retina

def mask(x, y, x_c=0., y_c=0., scale=.25, method='exp'):
    R2 = (x-x_c)**2/(scale*4/3)**2 + (y-y_c)**2/scale**2
    if method == 'gaussian': return np.exp(-.5*R2)
    if method == 'exp': return np.exp(-np.sqrt(R2))
    if method == 'scalefree': return 1./(R2+ .01)


def saliency(image, image2=None):
    """
    dummy saliency map:
     - derive

    """
    return scipy.ndimage.laplace(image)**2


def center_saliency(x, y, energy):
    """
    dummy saliency map:
     - derive

    """
    energy /= energy.sum()
    return np.sum(x[:]*energy[:]), np.sum(y[:]*energy[:])


if __name__ == '__main__':
    # neural parameters
    spike = 255*np.ones(45) # that's a crude spike!
    quant = 100
    hist = np.ones(quant) / quant
    rate = 0.01
    adaptive = True

    # initialize camera
    mode_num = 0
    device_num = 0
    num_buffers = 32

    snapshotTime = time.time()

    camera = cam_iface.Camera(device_num, num_buffers, mode_num)
    camera.start_camera()
    print ' Startup time ', (time.time() - snapshotTime)*1000, ' ms'
    snapshotTime = time.time()

    # initialize display
    w = window.Window(visible=False, resizable=True)
    arr = convert(np.asarray(camera.grab_next_frame_blocking()))
    im = retina(arr)
    aii = ArrayInterfaceImage(calibrate(im))
    size = im.shape
# 
#     img = pyglet.image.load('/Users/lup/Desktop/ball.png')
#     sprite = pyglet.sprite.Sprite(img)

    from scipy import mgrid
    x, y = mgrid[-1:1:1j*size[0],-1:1:1j*size[1]]
#     global x_c, y_c
    x_c, y_c = 0., 0.

    img = aii.texture
#     checks = image.create(32, 32, image.CheckerImagePattern())
#     background = image.TileableTexture.create_for_image(checks)

    w.width = img.width
    w.height = img.height
    w.set_visible()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    while not w.has_exit:
        w.dispatch_events()
        x_c_v, y_c_w, dot = int((x_c+1.)/2*w.width), int((y_c+1.)/2*w.height), 10
        vertices = [
                x_c_v+dot, y_c_w+dot,
                x_c_v+dot, y_c_w-dot,
                x_c_v-dot, y_c_w-dot,
                x_c_v-dot, y_c_w+dot ]
        vertices_gl = (GLfloat * len(vertices))(*vertices)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 2)

        graphics.draw(1, GL_POINTS, ('v2i', (int((x_c+1.)/2*size[0]), int((y_c+1.)/2*size[1]))))
#         background.blit_tiled(0, 0, 0, w.width, w.height)

        im = retina(convert(np.asarray(camera.grab_next_frame_blocking())))
        mask_ = mask(x, y, x_c, y_c)
        im *= mask_
        x_c, y_c = center_saliency(x, y, saliency(im))

#         print x_c, y_c
#         sprite.draw()
        im[mask_>.9] = 1 # make the central spot
        aii.view_new_array(calibrate(im)) # switch ArrayInterfaceImage to view the new array
        img.blit(0, 0, 0)
        w.flip()
#
#     @w.event
#     def on_draw():
#         w.dispatch_events()
# #         background.blit_tiled(0, 0, 0, w.width, w.height)
#         img.blit(0, 0, 0)
#         w.flip()
# 
#         im = retina(convert(np.asarray(camera.grab_next_frame_blocking())))
#         im *= mask(x, y, x_c, y_c)
#         aii.view_new_array(calibrate(im)) # switch ArrayInterfaceImage to view the new array
#         x_c, y_c = center_saliency(x, y, saliency(im))
#         graphics.draw(1, GL_POINTS, ('v2i', (int((x_c+1.)/2*size[0]), int((y_c+1.)/2*size[1]))))
#         print x_c, y_c
# #         label.draw()
# 
#     pyglet.app.run()

