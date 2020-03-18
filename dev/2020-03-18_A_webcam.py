import cv2
video_capture = cv2.VideoCapture(0)

import time
while True:
    tic = time.time()

    # Grab a single frame of video
    ret, frame = video_capture.read()# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]# Find all the faces in the current frame of video
    #cv2.imshow('Video', frame)# Hit ‘q’ on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    toc = time.time()
    print(f'FPS:{1/(toc-tic):.3f}')
