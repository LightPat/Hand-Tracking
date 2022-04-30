# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:55:27 2022

@author: Patrick Seeman
"""

import cv2
import mediapipe as mp
import time
import math
import screen_brightness_control as sbc

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
                      max_num_hands=1,
                      min_detection_confidence=0.3,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
 
# get the brightness of the primary display
primary_brightness = sbc.get_brightness(display=0)
print(primary_brightness)

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
    
    distThumbIndex = 0
    startPoint = (0,0)
    
    if handPosition.multi_hand_landmarks:
        for landmarkList in handPosition.multi_hand_landmarks:
            for ID, lm in enumerate(landmarkList.landmark):
                # Height, width, color
                h, w, c = frame.shape
                # Scale coordinates according to size of frame
                cx, cy = int(lm.x *w), int(lm.y*h)
                # ID 8 is the tip of the index finger
                # ID 4 is the tip of the thumb
                
                if ID == 4 or ID == 8:
                    cv2.circle(frame, (cx,cy), 3, (0,255,0), cv2.FILLED)
                    if startPoint != (0,0):
                        # cv2.line(frame, startPoint, (cx, cy), (0,0,0), 3)
                        distThumbIndex = math.dist(startPoint, (cx, cy))
                    startPoint = (cx, cy)
                else:
                    cv2.circle(frame, (cx,cy), 3, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, landmarkList, mpHands.HAND_CONNECTIONS)
    
    if (distThumbIndex != 0):        
        sbc.set_brightness(distThumbIndex/240*100, display=0)
    
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