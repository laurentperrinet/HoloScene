

import time
import zmq

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s …" % request)
    socket.send(b"GO!")

    #  Get the reply.
    message = socket.recv()
    # https://mkyong.com/python/python-3-convert-string-to-bytes/
    print("Received reply %s [ %s ]" % (request, message.decode()))
    t, b, l, r = message.decode().split(', ')
    t, b, l, r = int(t), int(b), int(l), int(r)
    print(f't, b, l, r = {t}, {b}, {l}, {r}')
