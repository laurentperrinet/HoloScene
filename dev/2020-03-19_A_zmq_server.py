import cv2
video_capture = cv2.VideoCapture(0)
N_X, N_Y = 320, 180
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, N_X)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, N_Y)

import dlib
import numpy as np

def grab(DS=4):
    ret, frame = video_capture.read()
    return frame[:, :, ::-1]

RESOLUTION = 1000

class FaceExtractor:
    def __init__(self, N_X, N_Y):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        # http://dlib.net/face_alignment.py.html
        self.predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        self.S = (N_X**2 + N_Y**2)

    def get_bbox(self, frame):
        N_X, N_Y, three = frame.shape
        dets = self.detector(frame, 1)
        if len(dets) > 0:
            bbox = dets[0]
            shape = self.predictor(frame, bbox)
            # [(p.x, p.y) for p in shape.parts()]
            coords = np.array([[p.x, p.y] for p in shape.parts()])
            # t, b, l, r = bbox.top(), bbox.bottom(), bbox.left(), bbox.right()
            t, b, l, r = coords[:, 1].min(), coords[:, 1].max(), coords[:, 0].min(), coords[:, 0].max()
            return t, b, l, r
        else:
            return 0, 0, 0, 0

    def center_size(self, t, b, l, r):
        return t + (b-t)/2, l + (r-l)/2, (b-t)**2 + (r-l)**2

    def center_normalized(self, x, y, s):
        """
         (0, 0) ----------> Y (0, 1)
         |
         |     image coordinates
         |
         V

         X (1, 0)

         """


        return x/N_X, y/N_Y/2, np.sqrt(s/self.S)

f = FaceExtractor(N_X, N_Y)

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

rgb_frame = grab()
t, b, l, r = f.get_bbox(rgb_frame)

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)
    tic = time.time()

    if not (t==0 and b==0 and l==0 and r==0):
        x, y, s = f.center_size(t, b, l, r)
        print(f'x, y, s = {x:.1f}, {y:.1f}, {s:.1f}')
        x, y, s = f.center_normalized(x, y, s)
        print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')
        x, y, s = int(x*RESOLUTION), int(y*RESOLUTION), int(s*RESOLUTION)
        #  Send reply back to client
        socket.send(f"{x}, {y}, {s}".encode())
    else:
        socket.send("ERROR".encode())

    rgb_frame = grab()
    t, b, l, r = f.get_bbox(rgb_frame)

    toc = time.time()
    print(f'FPS:{1/(toc-tic):.1f}')
