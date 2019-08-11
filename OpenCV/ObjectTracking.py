__author__ = 'Soniya Rode'
'''
Program to track blue, red and green color objects in a video input
'''
import cv2
import numpy as np

#Take video input from webcam
capture = cv2.VideoCapture(0)

while(1):

    # For each frame, .read() return true/false and the frame read.
    ret, frame = capture.read()

    # Convert BGR Image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of a color in HSV,
    # given range for blue color
    startBlue = np.array([110,50,50])
    endBlue= np.array([130,255,255])
    # given range for green color
    startGreen = np.array([50, 50, 120])
    endGreen = np.array([70, 255, 255])
    # given range for red color
    startRed = np.array([169, 100, 100], dtype=np.uint8)
    endRed= np.array([189, 255, 255], dtype=np.uint8)

    # Create a mask by thresholding the image
    mask1 = cv2.inRange(hsv,startBlue,endBlue)
    mask2 = cv2.inRange(hsv,startGreen, endGreen)
    mask3=cv2.inRange(hsv,startRed,endRed)

    mask=mask1+mask2+mask3

    # Do Bitwise-AND operation on the  original image with the mask to get the colored object
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    #press ESC to exit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
