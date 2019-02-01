#!/usr/bin/env python
# -*- coding:utf-8 -*-
import picamera
import picamera.array
import cv2
import numpy as np

# ----RGB THRESHOLD----
def get_th_rgb(img_blur,r1,g1,b1,r2,g2,b2):
    img_rgb = img_blur.copy()
    rgb_min = np.array([r1,g1,b1], np.uint8)
    rgb_max = np.array([r2,g2,b2], np.uint8)
    th      = cv2.inRange(img_rgb, rgb_min, rgb_max)
    return th

# ----HSV THRESHOLD----
def get_th_hsv(img_blur,h1,s1,v1,h2,s2,v2):
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    hsv_min = np.array([h1,s1,v1], np.uint8)
    hsv_max = np.array([h2,s2,v2], np.uint8)
    th      = cv2.inRange(img_hsv, hsv_min, hsv_max)
    return th

# ----ADAPTIVE THRESHOLD----
def get_th_ad(img_gray, Block_Size, C):
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block_Size, C) #
    return th

# ----SYNTHESIS THRESHOLD----
def get_th(img, val):
    img_blur   = cv2.GaussianBlur(img, (5,5), 0)
    img_gray   = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    th1 = get_th_rgb(img_blur,val[0],val[1],val[2],val[3],val[4],val[5])   # rgb
    th2 = get_th_hsv(img_blur,val[6],val[7],val[8],val[9],val[10],val[11]) # hsv
    th3 = get_th_ad(img_gray,2*val[12]+3,val[13])                          # ad
    th = cv2.bitwise_and(th1,cv2.bitwise_and(th2,th3))                     # synthesis
    return th

# ------------------------------------------------------------------------------------------

# function for event
def nothing(x):
    pass

# Create a window
STANDARD_SIZE = (512,300)
WINDOW_NAME   = 'image'
cv2.namedWindow(WINDOW_NAME)

# create trackbars and switch
trackbar_names = ['R1','G1','B1','R2','G2','B2','H1','S1','V1','H2','S2','V2','BS','C']
initial_values = [0,0,0,255,255,255,0,0,0,255,255,255,0,0]
for name,val in zip(trackbar_names,initial_values):
    cv2.createTrackbar(name,WINDOW_NAME,val,255,nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch,WINDOW_NAME,0,1,nothing)


with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (320,240)
        camera.framerate  = 15
        camera.awb_mode   = 'flourescent'
        
        while True:
            samera.capture(stream, 'bgr', use_video_port=Ture)
            img      = cv2.resize(stream.array, STANDARD_SIZE)
            img_blur = cv2.GaussianBlur(img, (5,5), 0)

            # get current positions of trackbars
            val_list = [cv2.getTrackbarPos(name,WINDOW_NAME) for name in trackbar_names]
            s        = cv2.getTrackbarPos(switch,WINDOW_NAME)  
            
            # show img(0) or threshold img(1)
            if s == 0:
                th = img
            else:
                th = get_th(img, val_list)
            cv2.imshow('img',th)
            
            # wait key input and if key=27(Esc) => break
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            # reset stream
            stream.seek(0)
            stream.truncate()
        
        # Close all windows
        cv2.destroyAllWindows()
        # print out last values
        print(val_list[0:6])
        print(val_list[6:12])
        print([2*val_list[12]+3, val_list[13]])
        