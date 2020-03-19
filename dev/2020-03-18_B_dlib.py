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

import dlib
hogFaceDetector = dlib.get_frontal_face_detector()

# http://dlib.net/face_detector.py.html
win = dlib.image_window()
# http://dlib.net/face_landmark_detection.py.html
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# http://dlib.net/face_alignment.py.html
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

import time
while True:
    tic = time.time()

    # Grab a single frame of video
    rgb_frame = grab()

    win.clear_overlay()


    win.set_image(rgb_frame)

    faceRects = hogFaceDetector(rgb_frame, 0)
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        shape = predictor(rgb_frame, faceRect)
        print(shape)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
    win.add_overlay(faceRects)
    # dlib.hit_enter_to_continue()

    #cv2.imshow('Video', frame)# Hit ‘q’ on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    toc = time.time()
    print(f'FPS:{1/(toc-tic):.3f}, num faces = {len(faceRects)}')
