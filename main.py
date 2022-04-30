# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:55:27 2022

@author: Patrick Seeman
"""

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Hand Tracking")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Hand Tracking", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()