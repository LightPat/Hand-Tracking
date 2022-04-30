# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:55:27 2022

@author: Patrick Seeman
"""

import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)
# Create window for frames from webcam to be displayed in
cv2.namedWindow("Hand Tracking")
# Used for naming images if we save frames using the SPACE key
img_counter = 0
# Used for calculating FPS
cTime = 0
pTime = 0

# Create hand tracking model
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    # Read in an image from the webcam
    success, frame = cam.read()
    if not success:
        print("failed to grab frame")
        break

    # Calculate and display frames per second
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    handPosition = hands.process(frame)
    
    if handPosition.multi_hand_landmarks:
        for landmarkList in handPosition.multi_hand_landmarks:
            for ID, lm in enumerate(landmarkList.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)
                
            mpDraw.draw_landmarks(frame, landmarkList, mpHands.HAND_CONNECTIONS)
    
    # Display in window
    cv2.imshow("Hand Tracking", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed, save frame to disk
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()