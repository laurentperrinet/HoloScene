import cv2
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)


import dlib
import numpy as np

def grab(DS=1):
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

    def center_size(self, t, b, l, r):
        return t + (b-t)/2, l + (r-l)/2, (b-t)**2 + (r-l)**2

    def center_normalized(self, x, y, s):
        return 2*(x - N_X/2)/N_X, 2*(y - N_Y/2)/N_Y, s / (N_X**2 + N_Y**2)

f = FaceExtractor(N_X, N_Y)

import time
while True:
    tic = time.time()

    # Grab a single frame of video
    rgb_frame = grab()

    # detect face
    t, b, l, r = f.get_bbox(rgb_frame)
    toc = time.time()

    print(f'FPS:{1/(toc-tic):.1f}')
    print(f't, b, l, r = {t}, {b}, {l}, {r}')
    if not (t==0 and b==0 and l==0 and r==0):
        x, y, s = f.center_size(t, b, l, r)
        print(f'x, y, s = {x:.1f}, {y:.1f}, {s:.1f}')
        x, y, s = f.center_normalized(x, y, s)
        print(f'x, y, s (norm) = {x:.3f}, {y:.3f}, {s:.3f}')
