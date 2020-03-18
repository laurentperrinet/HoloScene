import cv2
video_capture = cv2.VideoCapture(0)

import dlib
hogFaceDetector = dlib.get_frontal_face_detector()
DS = 4

import time
while True:
    tic = time.time()

    # Grab a single frame of video
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[::DS, ::DS, ::-1]# Find all the faces in the current frame of video

    faceRects = hogFaceDetector(rgb_frame, 0)
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()

    #cv2.imshow('Video', frame)# Hit ‘q’ on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    toc = time.time()
    print(f'FPS:{1/(toc-tic):.3f}, num faces = {len(faceRects)}')
