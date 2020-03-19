# -*- coding: utf-8 -*-
import time
import zmq

zmqcontext = zmq.Context()

#  Socket to talk to server
print("Connecting to server…")
socket = zmqcontext.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

message = ""
#  Get the reply.
while True:
    tic = time.time()
    print("Sending request … GO!")
    socket.send(b"GO!")

    message = socket.recv()
    message = message.decode()
    print(message)

    if message == "ERROR":
        break
        sys.exit()

    toc = time.time()

    print(f'FPS:{1/(toc-tic):.1f}')
