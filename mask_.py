# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 00:45:19 2022

@author: ess305
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_mask(hct, ep) -> list:
    #1630
    # 從388nm的圖 擷取mask的圓形半徑資訊
    mask_img = hct
    img = ep

    mask_img_gray = cv2.cvtColor(mask_img,cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    low_threshold = 15
    high_threshold = 30
    edges = cv2.Canny(mask_img_gray , low_threshold , high_threshold)

    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1000,100,param1 = 100,param2 = 30,minRadius = 1000,maxRadius = 2000)

    circle = circles[0, :, :] 


    mask_img_BGR = cv2.cvtColor(mask_img_gray,cv2.COLOR_GRAY2BGR)
    img_circle = cv2.circle(mask_img_BGR,(4500,4500),3600,(0,0,255),-1)
    plt.imshow(img_circle)

    shape = (9081,9081,3)
    origin = np.zeros(shape,np.uint8)
    origin_circle = cv2.circle(origin,(4500,4500),3600,(255,255,255),-1)
    origin_circle = cv2.cvtColor(origin_circle,cv2.COLOR_RGB2GRAY)
    origin_circle[origin_circle>0] = 1
    #plt.imshow(origin_circle,cmap = 'Greys_r')
    #plt.show()

    masked = origin_circle*img_gray
    # plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
    # plt.imshow(masked , cmap = 'Greys_r')
    # plt.show()
    # plt.imsave(r"D:\code\blood\data\masked_1853.jpg",masked,cmap = 'Greys_r')

    return masked