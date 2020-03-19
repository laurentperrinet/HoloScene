# -*- coding: utf-8 -*-
import time
import sys
import zmq
import numpy as np

RESOLUTION = 1000
z0 = .65 # in meter
s0 = .15 # normalized unit
VA = 45 # visual angle (in degrees) of the camera

#  Socket to talk to server
zmqcontext = zmq.Context()
print("Connecting to server…")
socket = zmqcontext.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

message = ""
#  Get the reply.
while True:
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

    x, y, s = message.split(', ')
    x, y, s = int(x), int(y), int(s) # str > int
    x, y, s = x/RESOLUTION, y/RESOLUTION, s/RESOLUTION
    x, y, s = 2*x-1, 2*y-1, s
    # print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')

    z = z0 * np.tan(s0 / 2 * VA) / np.tan(s / 2 * VA)
    x =  z * np.tan(x * VA)
    y =  z * np.tan(y * VA)
    print(f'x, y, z (Eye) = {x:.3f}, {y:.3f}, {z:.3f}')

    toc = time.time()

    print(f'FPS:{1/(toc-tic):.1f}')
