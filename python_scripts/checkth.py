#!/usr/bin/env python
# -*- coding:utf-8 -*-
import picamera
import picamera.array
import cv2
import numpy as np
from threshold.py import *


with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (320,240)
        camera.framerate  = 15
        #camera.awb_mode   = 'flourescent'
        
        while(1):
            camera.capture(stream, 'bgr', use_video_port=True)
            img = stream.array
            cv2.imshow(img)
            
            th       = get_th(img)         # 白っぽい部分を取得
            cnt      = get_max_cnt(th)     # 最大部分を囲む矩形
            target   = get_target(img,cnt) # 的の板を正面にする
            img_rect = cv2.drawContours(img.copy(), [cnt], 0, (255,0,0), 12)
    
            # 画像を表示する
            cv2.imshow('th', th)
            cv2.imshow('img_rect', img_rect)
            cv2.imshow('target', target)

            # キー入力を1ms待って、kが27(Esc)だったらbreakする
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            # streamをリセット
            stream.seek(0)
            stream.truncate()
        
        cv2.destroyAllWindows()
