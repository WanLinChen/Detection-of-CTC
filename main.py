# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:00:55 2022

@author: chris
"""
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
# import seaborn as sns
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture
from threshold import compute_threshold,compare_structure
from scipy.spatial import distance
from skimage import measure,color
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops, regionprops_table
from mask_ import get_mask

#%% store the image in one folder "data"
DATADIRECTORY = r"/data/"
current_dir =  os.path.abspath(os.path.dirname(__file__))
img_list = glob.glob(os.path.join(current_dir+DATADIRECTORY, "*.jpg"))
#%% input images

# print(img_list)
for i in img_list:
    if 'F_' and '_0.jpg' in i:
        img_col1 = cv2.imread(i)
    elif 'F_' and '_1.jpg' in i:
        img_col2 = cv2.imread(i)
    elif 'F_' and '_3.jpg' in i:
        img_col3 = cv2.imread(i)
    else:
        img_white = cv2.imread(i)

#input Epcam (after masked)
img = get_mask(img_col1, img_col2)
img1 = img
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur_col1 = cv2.GaussianBlur(img_col1,(5,5),0)
#%% #find local maximum and scatter light spot

thr = compute_threshold(img1)
xy = peak_local_max(img,min_distance = 20, threshold_abs = thr)

#%%
SIM = []
DTC = []
IOU = []
m=[]
light_mean = []
y_pred=[]

for n in range(len(xy)):
    #if  n==0  : #  
        img_sep = img1[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        img_sep1 = img1[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        
        img_sepb_col1 = blur_col1[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        img_sep_col2 = img_col2[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        img_sep_col3 = img_col3[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        img_sep_white = img_white[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
        
        img_seperation = img[xy[n,0]-20:xy[n,0]+20,xy[n,1]-20:xy[n,1]+20]
       
        img_seperation_BGR = cv2.cvtColor(img_seperation,cv2.COLOR_GRAY2BGR)
        
        cx = 0
        cy = 0
        s = np.sum(img_seperation)
       
        #畫出輪廓
        ret,binary = cv2.threshold(img_seperation,25,255,cv2.THRESH_BINARY)
        _,contours , hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

       
        #求出輪廓重心座標
        for i in range(40):
           cx += i*(np.sum(img_seperation[i,:]))/s
           cy += i*(np.sum(img_seperation[:,i]))/s
       
        #use Gaussian mixture model decide thr of 488nm
        GMM =GaussianMixture(n_components=2)
        GMM.fit_predict(img_seperation)
        thr_488 = round(np.mean(GMM.means_),3) 
        
        #use cv2.thr
        ret1,thresh = cv2.threshold(img_seperation,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        tmp=img_seperation_BGR[:,:,0]
        tmp[tmp<ret1] = 0
        tmp[tmp>ret1] = 250
        img_seperation_BGR[:,:,0]=tmp
        img_seperation_BGR[:,:,1]=0
        img_seperation_BGR[:,:,2]=0
        
        #calulate the mean of light part
        light_488 = img_seperation.copy()
        
        light_488[light_488<ret1] = 0

        
        #calulate the area of 488nm
        list_488 = img_seperation_BGR[:,:,0].flatten()
        num_488 = 0
        
        for i in range(1600):
            if list_488[i] == 250:
                num_488 += 1
       
        area = num_488
        radius = math.sqrt(area/np.pi)
        diameter = 2*radius
       
        #在全黑的圖上畫出對應的圓
        img_black = np.zeros((40,40,3),np.uint8)
        img_black_circle = cv2.circle(img_black,(round(cx),round(cy)),round(radius),(255,255,255),-1)
 
        #SSIM
        img_seperation[img_seperation<thr] = 0
        img_seperation[img_seperation>thr] = 255
        
        img_black_circle = cv2.cvtColor(img_black_circle,cv2.COLOR_BGR2GRAY)
        ssim = compare_structure(img_seperation,img_black_circle)
         
     
        SIM.append(ssim) 
        
        
        light = []
        list_light_488 = light_488.flatten()
        for j in range(1600):
            if list_light_488[j] >0:
                light.append(list_light_488[j])
        light_488_mean = sum(light)/len(light)
        
        #若要增加圈選條件 則加在這裡
        #if  6<= round(diameter) <= 16 and 0.572<= round(ssim,3) <= 0.783 and 34.317 <= round(light_488_mean,3) <= 231.625:
        m.append(n)
        print(n)
        print("Diameter: "+ str(round(diameter)))
        print("Mean: "+ str(light_488_mean))
        print("SSIM: "+ str(round(ssim,3)))
        print("-----------------------------------")
     
        
        #erosion & dilation of 365nm
        kernel=np.ones((3,3),np.uint8)
        erosion_365 = cv2.erode(img_sepb_col1 ,kernel, iterations=1)
        dilation_365_gray =  cv2.cvtColor(erosion_365,cv2.COLOR_BGR2GRAY)
     
         
        GMM =GaussianMixture(n_components=2)
        GMM.fit(dilation_365_gray)
        thr_365 = round(np.mean(GMM.means_),3) 
     
         
        #label 365nm
        labels = measure.label(dilation_365_gray>thr_365,connectivity=2)          
        regions = measure.regionprops(labels)
        props = regionprops_table(labels, properties=('centroid','area'))
        
        img_center = (20,20)
        dictionaries = []
        for q in range(len(regions)):
             c_0 = props.get('centroid-0')
             c_1 = props.get('centroid-1')
             a = props.get('area')
             cx = c_0[q]
             cy = c_1[q]
             area = a[q]
             contour_center = (cx,cy)
             
             distances_to_center = (distance.euclidean(img_center,contour_center))
             dictionaries.append({'contour':q,'center':contour_center,'distance_to_center': distances_to_center,'area':area})
         
        sorted_distance = sorted(dictionaries,key=lambda i: i['distance_to_center'])
         
        DTC.append(sorted_distance[0]['distance_to_center'])
    
    
        area_img_cc = sorted_distance[0]['area'] 
     
        img_cc = labels==sorted_distance[0]['contour']+1
        img_cc_gray = img_cc*255
        img_cc_gray = img_cc_gray.astype('uint8')
        img_cc_BGR =  cv2.cvtColor(img_cc_gray,cv2.COLOR_GRAY2BGR)
        img_cc_BGR[:,:,0] = 0
        img_cc_BGR[:,:,1] = 0 
        img_cc_BGR[:,:,2] = img_cc_gray
         
        
        #plus 488nm & 365nm contour together
        img_overlap = img_seperation_BGR + img_cc_BGR
        img_overlap_gray = cv2.cvtColor(img_overlap,cv2.COLOR_BGR2GRAY)
        
        
        #find overlap part
        num_intersection = 0
        num_inter = 0
        
        dil_sum = img_overlap_gray.flatten()
        
        for i in range(1600):
            if dil_sum[i] > 0:
                num_intersection += 1
                
        for i in range(1600):
            if dil_sum[i] > 100:
                num_inter += 1
        iou_1 = num_inter/(num_intersection) #change to area_img_cc
         
        IOU.append(iou_1)
    
        
        #show the information
        alpha = 0.35
        #488nm image show
        front = img_sep.copy()
        img_488_show = cv2.addWeighted(front, 1-alpha, img_seperation_BGR, alpha, 0)
        
        #365nm image show
        img_365_show = cv2.addWeighted(img_sepb_col1, 1-alpha, img_cc_BGR, alpha, 0)
         
        #365+630 overlap image show
        img_365630_over =  cv2.addWeighted(img_sep_col3, 1-alpha, img_seperation_BGR, alpha, 0)
         
     
        #show hand label image information
         
        plt.figure(figsize = (10,5))
         
        plt.subplot(2,5,1)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('Epcam_raw',{'fontsize':10})
        plt.imshow(img_sep)
           
        plt.subplot(2,5,2)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('Epcam',{'fontsize':10})
        plt.imshow(img_488_show)
     
        plt.subplot(2,5,3)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('hoechst',{'fontsize':10})
        plt.imshow(img_365_show)
           
        plt.subplot(2,5,4)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('overlap',{'fontsize':10})
        plt.imshow(img_overlap)
         
        plt.subplot(2,5,6)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('white',{'fontsize':10})
        plt.imshow(img_sep_white)  
        
        
        plt.subplot(2,5,7)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('CD45',{'fontsize':10})
        plt.imshow(img_sep_col3)  
        
        plt.subplot(2,5,8)
        plt.tick_params(left = False,right = False,labelleft = False,labelbottom = False,bottom = False)
        plt.title('Epcam & CD45 overlap',{'fontsize':10})
        plt.imshow(img_365630_over)  
         
        plt.text(50,30,str(n)+'_diameter: '+str(round(diameter,3)),{'fontsize':10})
        plt.text(50,20,str(n)+'_lightness: '+str(round(light_488_mean,3)),{'fontsize':10})
        plt.text(50,10,str(n)+'_SSIM: '+str(round(ssim,3)),{'fontsize':10}) 
        plt.text(100,30,str(n)+'_iou: '+ str(round(iou_1,3)),{'fontsize':10})
        plt.text(100,20,str(n)+'_distance: '+ str(round(sorted_distance[0]['distance_to_center'],3)),{'fontsize':10})

        plt.savefig('1715_'+ str(n) +'.jpg',dpi=1200,transparent=True)