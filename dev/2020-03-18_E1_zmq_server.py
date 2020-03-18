import cv2
video_capture = cv2.VideoCapture(0)


import dlib
import numpy as np

def grab(DS=4):
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    return frame[::DS, ::DS, ::-1]# Find all the faces in the current frame of video

rgb_frame = grab()
N_X, N_Y, three = rgb_frame.shape
print('N_X, N_Y, three =', N_X, N_Y, three)

# https://github.com/laurentperrinet/LeCheapEyeTracker/blob/3d203f4f301695a4fbb07f16c7c289abac514348/src/LeCheapEyeTracker/EyeTrackerServer.py
class FaceExtractor:
    def __init__(self, N_X, N_Y):
        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def get_bbox(self, frame):
        N_X, N_Y, three = frame.shape
        dets = self.detector(frame, 1)
        if len(dets) >0:
            bbox = dets[0]
            t, b, l, r = bbox.top(), bbox.bottom(), bbox.left(), bbox.right()
            return t, b, l, r
        else:
            return 0, 0, 0, 0

f = FaceExtractor(N_X, N_Y)

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    tic = time.time()
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    # Grab a single frame of video
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = grab()

    # detect face
    t, b, l, r = f.get_bbox(rgb_frame)
    toc = time.time()

    print(f'FPS:{1/(toc-tic):.1f}')
    print(f't, b, l, r = {t}, {b}, {l}, {r}')
    # send_array(self.out_socket, data, dtype=self.dtype)

    #  Send reply back to client
    socket.send(f"{t}, {b}, {l}, {r}".encode())
