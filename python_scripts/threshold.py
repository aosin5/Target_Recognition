# -*- coding: utf-8 -*-
import numpy as np
import cv2


def get_th_rgb(img,r1,g1,b1,r2,g2,b2):
    img_rgb = img.copy()                            # rgbスケール
    rgb_min = np.array([r1,g1,b1], np.uint8)
    rgb_max = np.array([r2,g2,b2], np.uint8)
    th      = cv2.inRange(img_rgb, rgb_min, rgb_max)
    return th


def get_th_hsv(img,h1,s1,v1,h2,s2,v2):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # hsvスケール
    hsv_min = np.array([h1,s1,v1], np.uint8)
    hsv_max = np.array([h2,s2,v2], np.uint8)
    th      = cv2.inRange(img_hsv, hsv_min, hsv_max)
    return th


def get_th_ad1(img_gray, Block_Size, C):
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block_Size, C)
    return th


def get_th_ad2(img_gray, Block_Size, C):
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Block_Size, C)
    return th


def get_th(img):
    # グレースケール画像の準備
    img_blur   = cv2.GaussianBlur(img, (5,5), 0)
    img_gray   = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    th1 = get_th_rgb(img_blur,100,100,100,255,255,255) # rgb
    th2 = get_th_hsv(img_blur,0,0,100,200,35,255)      # hsv
    th3 = get_th_ad1(img_gray,39,1)                    # ad1
    th4 = get_th_ad2(img_gray,39,1)                    # ad2

    th = cv2.bitwise_and(th1,th2)
    th = cv2.bitwise_and(th, th3)
    th = cv2.bitwise_and(th, th4)
    return th


def get_contours(th):
    _, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_max_cnt(th):
    # 輪郭を取得
    contours = get_contours(th)
    if(len(contours)==0):
        width,height = th.shape
        return np.array([[[0,0]],[[0,width]],[[height,width]],[[height,0]]])
    
    # 最大輪郭を取得
    cnt_list = np.array([cv2.moments(cnt)['m00'] for cnt in contours]) # 各cntの面積リスト
    index    = np.argmax(cnt_list)  # 面積最大の指標
    max_cnt  = contours[index]      # 最大cntを取得
    
    # 最大輪郭の近似(滑らかにする)
    epsilon = 0.1*cv2.arcLength(max_cnt,True)
    max_cnt = cv2.approxPolyDP(max_cnt,epsilon,True)
    return max_cnt


def get_target(img,cnt):
    if(len(cnt)==4):
        # 頂点座標の整理
        cnt = cnt.tolist()
        cnt.sort(key=lambda x:x[0][0]+x[0][1])
        cnt = np.array(cnt)
        # 切り出し箇所
        pts1 = np.float32([[cnt[0,0,0],cnt[0,0,1]],  # 左上点
                           [cnt[1,0,0],cnt[1,0,1]],  # 右上点
                           [cnt[2,0,0],cnt[2,0,1]],  # 左下点
                           [cnt[3,0,0],cnt[3,0,1]]]) # 右下点
        pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]]) # 切り出し後サイズ
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(300,300))
        return dst
    else:
        return np.zeros((300,300))