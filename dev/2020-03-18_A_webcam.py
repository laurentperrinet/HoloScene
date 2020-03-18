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

import time
while True:
    tic = time.time()
    rgb_frame = grab()

    #cv2.imshow('Video', frame)# Hit ‘q’ on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    toc = time.time()
    print(f'FPS:{1/(toc-tic):.3f}')
