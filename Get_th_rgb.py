#!/usr/bin/env python
# -*- coding:utf-8 -*-
import picamera
import picamera.array
import cv2
import numpy as np
from threshold.py import *


def nothing(x):
    pass

# Create a black image, a window
STANDARD_SIZE = (512,300)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R1','image',0,255,nothing)
cv2.createTrackbar('G1','image',0,255,nothing)
cv2.createTrackbar('B1','image',0,255,nothing)
cv2.createTrackbar('R2','image',255,255,nothing)
cv2.createTrackbar('G2','image',255,255,nothing)
cv2.createTrackbar('B2','image',255,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

# 0 => IVcan, 1 => Webカメラ
cap = cv2.VideoCapture(1)


with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (320,240)
        camera.framerate  = 15
        camera.awb_mode   = 'flourescent'
        
        while True:
            samera.capture(stream, 'bgr', use_video_port=Ture)
            img = stream.array
            cv2.imshow(img)
            
            img        = cv2.resize(img, STANDARD_SIZE)
            img_blur   = cv2.GaussianBlur(img, (5,5), 0)
            cv2.imshow('img',th)

            # get current positions of four trackbars
            r1 = cv2.getTrackbarPos('R1','image')
            g1 = cv2.getTrackbarPos('G1','image')
            b1 = cv2.getTrackbarPos('B1','image')
            r2 = cv2.getTrackbarPos('R2','image')
            g2 = cv2.getTrackbarPos('G2','image')
            b2 = cv2.getTrackbarPos('B2','image')    
            s  = cv2.getTrackbarPos(switch,'image')
        
            if s == 0:
                th = img
            else:
                th = get_th_rgb(img_blur,r1,g1,b1,r2,g2,b2)
            
            # キー入力を1ms待って、kが27(Esc)だったらbreakする
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            # streamをリセット
            stream.seek(0)
            stream.truncate()
        
        cv2.destroyAllWindows()
