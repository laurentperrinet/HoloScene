# -*- coding: utf-8 -*-
import numpy as np
import time

RESOLUTION = 1000

n_x = 1600
n_y = 800

# n_x = 800
# n_y = 550

import zmq
import sys
from vispy import app, gloo, visuals
from vispy.geometry import create_box
from vispy.visuals.transforms import MatrixTransform

class Canvas(app.Canvas):

    def __init__(self, n_x=n_x, n_y=n_y):
        # screen
        app.Canvas.__init__(self,
                            keys='interactive',
                            size=(n_x, n_y), fullscreen=True)
        self.n_x, self.n_y = n_x, n_y

        # capture
        self.zmqcontext = zmq.Context()

        #  Socket to talk to server
        print("Connecting to local server…")
        self.socket = self.zmqcontext.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        # scene
        vertices, faces, outline = create_box(width=1, height=1, depth=1,
                                              width_segments=4,
                                              height_segments=8,
                                              depth_segments=16)

        self.box = visuals.BoxVisual(width=1, height=1, depth=1,
                                     width_segments=4,
                                     height_segments=8,
                                     depth_segments=16,
                                     vertex_colors=vertices['color'],
                                     edge_color='b')


        self.z0 = .65 # in meter
        self.s0 = .15 # normalized unit
        #self.THETA = 90
        self.VA_X = 30 * np.pi/180 # vertical visual angle (in radians) of the camera
        self.VA_Y = 45 * np.pi/180 # horizontal visual angle (in radians) of the camera
        self.SCALE = .2 * np.sqrt(self.n_x**2 + self.n_y**2)
        self.theta = 0
        self.phi = 0
        self.rotation_speed = .5
        # straight ahead
        self.x = 0
        self.y = 0
        self.z = self.z0

        self.transform = MatrixTransform()

        self.box.transform = self.transform
        self.show()

        self.timer = app.Timer(connect=self.rotate)
        self.timer.start(0.016)

    def rotate(self, event):
        # self.theta = self.x * self.THETA
        # self.phi = self.y * self.THETA

        # TODO:  make a random walk using a OU process
        self.theta += self.rotation_speed
        self.phi += self.rotation_speed
        self.transform.reset()
        self.transform.rotate(self.theta, (0, 0, 1))
        self.transform.rotate(self.phi, (0, 1, 0))
        scale = self.SCALE
        self.transform.scale((scale, scale, 0.001))
        self.transform.translate((self.n_x/2, self.n_y/2))
        self.update()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        self.box.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, ev):
        message = "ERROR"
        tic = time.time()
        #  Get the reply.
        while (message == "ERROR"):
            print("Sending request … GO!")
            self.socket.send(b"GO!")

            message = self.socket.recv()
            message = message.decode()
            print(message)

            if message == "ERROR":
                print(message)
                app.quit()
                sys.exit()

        x, y, s = message.split(', ')
        x, y, s = int(x), int(y), int(s) # str > int
        x, y, s = x/RESOLUTION, y/RESOLUTION, s/RESOLUTION
        x, y, s = x-.5, y-.5, s
        # print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')

        z = self.z0 * np.tan(self.s0 / 2 * self.VA) / np.tan(s / 2 * self.VA)
        z = self.z0 * self.s0 / s
        x = - z * np.tan(x * self.VA_X)
        y = - z * np.tan(y * self.VA_Y)

        print(f'x, y, z (Eye) = {x:.3f}, {y:.3f}, {z:.3f}')

        self.x, self.y, self.z = x, y, z
        gloo.clear(color='white', depth=True)
        self.box.draw()
        toc = time.time()

        print(f'FPS:{1/(toc-tic):.1f}')

win = Canvas()
app.run()
