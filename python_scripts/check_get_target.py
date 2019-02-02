#!/usr/bin/env python
# -*- coding:utf-8 -*-
import picamera
import picamera.array
import cv2
import numpy as np

# ========UTILITIES============================================================================

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
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block_Size, C)
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


# ----RETURN CONTOURS----
def get_contours(th):
    _, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ----RETURN LARGEST CNTOUR----
def get_max_cnt(th):
    contours = get_contours(th)
    if(len(contours)==0):
        width,height = th.shape
        return np.array([[[0,0]],[[0,width]],[[height,width]],[[height,0]]])
    
    # get largest contour
    cnt_list = np.array([cv2.moments(cnt)['m00'] for cnt in contours]) # cnt's area
    index    = np.argmax(cnt_list)  # index of largest area
    max_cnt  = contours[index]
    
    # smooth largest contour
    epsilon = 0.1*cv2.arcLength(max_cnt,True)
    max_cnt = cv2.approxPolyDP(max_cnt,epsilon,True)
    return max_cnt


# ----RETURN TARGET AREA (300*300)----
def get_target(img,cnt):
    if(len(cnt)==4):
        # sort vertex
        cnt = cnt.tolist()
        cnt.sort(key=lambda x:x[0][0]+x[0][1])
        cnt = np.array(cnt)
        # cut target
        pts1 = np.float32([[cnt[0,0,0],cnt[0,0,1]],  # left  up  
                           [cnt[1,0,0],cnt[1,0,1]],  # right up
                           [cnt[2,0,0],cnt[2,0,1]],  # left  down
                           [cnt[3,0,0],cnt[3,0,1]]]) # right down
        pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]]) # size of target
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(300,300))
        return dst
    else:
        return np.zeros((300,300))

    
# ==================================================================================================
    

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (320,240)
        camera.framerate  = 15
        #camera.awb_mode   = 'flourescent'
        val_list = [25,60,85,150,175,170, 95,75,90,165,170,145, 3,255]
        
        while(1):
            camera.capture(stream, 'bgr', use_video_port=True)
            img = stream.array
            cv2.imshow(img)

            th       = get_th(img,val_list)  # get threshold
            cnt      = get_max_cnt(th)       # get largest contour
            target   = get_target(img,cnt)   # cut target
            img_rect = cv2.drawContours(img.copy(), [cnt], 0, (255,0,0), 12) # draw rect
    
            # show image
            cv2.imshow('threshold', th)
            cv2.imshow('img_rect', img_rect)

            #  wait key input and if key=27(Esc) => break
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            # reset stream
            stream.seek(0)
            stream.truncate()
        
        cv2.destroyAllWindows()
