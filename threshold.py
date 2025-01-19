# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:21:26 2021

@author: ess
"""
import numpy as np
import cv2
from scipy import ndimage 
import statistics
from warnings import warn
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.dtype import dtype_range


def compute_threshold(masked):
    import cv2
    import os
    try:
        import numpy as np
        gpu = True
    except:
        import numpy as np
        pass

    # masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    image_stack = np.array(masked)
    image_stack = np.expand_dims(image_stack, 2)

    #path = r"D:\github\graph\lab\masked_1630"---------------------
    # data = os.listdir(path)
    
    # flag = 0
    # for img in data:
    #     temp = cv2.imread(path+'/masked_1853.jpg')#path+"/"+img
    #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #     if not flag:
    #         image_stack = np.array(temp)
    #         image_stack = np.expand_dims(image_stack, 2)
    #         flag = 1
    #     else:
    #         temp = np.array(temp)
    #         temp = np.expand_dims(temp, 2)
    #         image_stack = np.concatenate((image_stack, temp), 2)
    
    return np.quantile(image_stack, 0.5)+4*(np.quantile(image_stack, 0.75)-np.quantile(image_stack, 0.25))


def Covariance(x, y):
    xbar, ybar = x.mean(), y.mean()
    return (np.sum((x - xbar)*(y - ybar)))/(len(x)*len(x) - 1)


def compare_structure(img1,img2,data_range = None):
    
    if data_range is None:
        if img1.dtype != img2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[img1.dtype.type]
        data_range = dmax - dmin
        
    R = data_range
    K2 = 0.03
    C2 = (K2 * R) ** 2
    C3 = C2/2
   
    #standard deviation
    stdev1 = np.std(img1)
    stdev2 = np.std(img2)
    
       
    #covariances 
    #cov = np.cov(img1,img2)
    cov = Covariance(img1,img2)
    
    
    A1 = cov + C3
    B1 = stdev1 * stdev2 + C3
    S = A1/B1
      
    return S

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math
    import matplotlib.pyplot as plt

    brightness = 0
    contrast = -100 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)

    print("減少對比度 (白黑都接近灰，分不清楚): ")
    plt.imread(img)
    return plt.show()

def modify_contrast_and_brightness(img):
    # 公式： Out_img = alpha*(In_img) + beta
    # alpha: alpha參數 (>0)，表示放大的倍数 (通常介於 0.0 ~ 3.0之間)，能夠反應對比度
    # a>1時，影象對比度被放大， 0<a<1時 影象對比度被縮小。
    # beta:  beta参数，用來調節亮度
    # 常數項 beta 用於調節亮度，b>0 時亮度增強，b<0 時亮度降低。
    import matplotlib.pyplot as plt

    array_alpha = np.array([2.0]) # contrast 
    array_beta = np.array([0.0]) # brightness

    # add a beta value to every pixel 
    img = cv2.add(img, array_beta)                    

    # multiply every pixel value by alpha
    img = cv2.multiply(img, array_alpha)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255)

    print("增加對比度 - 網路上常見的方法 (但沒有實現黑的更黑這件事): ")
    plt.imread(img)
    return plt.show()




