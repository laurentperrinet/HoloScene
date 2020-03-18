import cv2
video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

import dlib
import numpy as np

def grab(DS=4):
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    return frame[::DS, ::DS, ::-1]# Find all the faces in the current frame of video

rgb_frame = grab()
N_X, N_Y, three = rgb_frame.shape
RESOLUTION = 1000
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
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    tic = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = grab()

    # detect face
    t, b, l, r = f.get_bbox(rgb_frame)

    if not (t==0 and b==0 and l==0 and r==0):
        x, y, s = f.center_size(t, b, l, r)
        print(f'x, y, s = {x:.1f}, {y:.1f}, {s:.1f}')
        x, y, s = f.center_normalized(x, y, s)
        print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')
        x, y, s = int(x*RESOLUTION), int(y*RESOLUTION), int(s*RESOLUTION)
        #  Send reply back to client
        socket.send(f"{x}, {y}, {s}".encode())

    toc = time.time()
    print(f'FPS:{1/(toc-tic):.1f}')
