#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:09:48 2018

@author: sanglinwei
"""
#import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import os

im=cv2.imread('/Users/sanglinwei/Downloads/big_data/pic_1.png')
im2=cv2.imread('/Users/sanglinwei/Downloads/big_data/pic_0.05.png')

'''
imgray1=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(imgray1,132,255,0)
ret,thresh2 = cv2.threshold(imgray2,132,255,0)

im3=thresh1-thresh2
plt.imshow(im3)
'''
'''
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(im2,kernel,iterations = 1)
dilation = cv2.dilate(im2,kernel,iterations = 1)
gradient = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, kernel)
plt.imshow(dilation)
#plt.imshow(erosion)
#plt.imshow(thresh)
'''
im1=im.copy()
imgray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(imgray1,127,255,0)
image1, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(0,len(contours)):
    if contours[i].size>500: #轮廓长度大于500个像素点，则记为目标轮廓
        cv2.drawContours(im1,contours,i,(255,255,255),-1)#-1可以填充轮廓内部
        edge=np.mat(contours[i])
#离散分布点区域
imgray = cv2.cvtColor(im,cv2.COLOR_BGRA2GRAY)
#imgray[imgray==0]=127
plt.imshow(imgray,cmap="gray")
ret,thresh=cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh,'gray')

kernel = np.ones((7,7),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#开运算
#closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel1,iterations = 1)
#闭运算，填充内部区域
#plt.imshow(closing,'gray')
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
# 距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离
# 个最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全
# 腐蚀。这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心􅑗􂅥像素点的
# 距离。根据各个像素点的距离值，设置为不同的灰度值。这样就完成了二值图像的距离变换
#cv2.distanceTransform(src, distanceType, maskSize)
dist_transform = cv2.distanceTransform(opening,1,5)
ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers1 = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers1+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers3 = cv2.watershed(im,markers)
im[markers3 == -1] = [255,0,0]

#故障区分：
'''
im[markers3==2]=[0,0,255]
im[markers3==3]=[0,0,255]
im[markers3==4]=[0,0,255]
'''
for i in range(2,markers3.max()+1):
    im[markers3==i]=[0,0,255]

for i in range(288):
    for j in range(432):
        if all(im[i,j]==[0,0,0]):
            im[i,j]=[0,127,0]
            
for i in range(288):
    for j in range(432):
        if all(im1[i,j]==[0,0,0]):
            im[i,j]=[255,0,0]            
            
#im[markers3==1]=[255,255,255]

#im[im==[0,0,0]]=67
#cv2.drawContours(im,markers3,1,(255,0,0),-1)
'''            
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
k=cv2.waitKey(0)&0xff
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    ccv2.imwrite('/Users/sanglinwei/Downloads/big_data/demo_1.png',im)
    cv2.destroyAllWindows()
'''
'''
cv2.namedWindow("image")
cv2.imshow('image', im)
cv2.waitKey(0) # close window when a key press is detected
cv2.destroyWindow('image')
cv2.waitKey(1)
'''
            
plt.imshow(opening,'gray')
# im,opening,sure_fg,unknown,opening,
cv2.imwrite('/Users/sanglinwei/Downloads/big_data/demo_1.png',im)

'''
plt.subplot(221),plt.imshow(opening,'gray')
plt.subplot(222),plt.imshow(closing,'gray')
plt.subplot(223),plt.imshow(unknown,'gray')
plt.subplot(224),plt.imshow(im,'gray')
'''
#imgray = cv2.cvtColor(im,cv2.COLOR_BGRA2GRAY)
#ret,thresh = cv2.threshold(imgray,132,255,0)
#plt.imshow(thresh)
'''
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(0,len(contours)):
    if contours[i].size>500: #轮廓长度大于500个像素点，则记为目标轮廓
        cv2.drawContours(im,contours,i,(255,0,0),2)#-1可以填充轮廓内部
        cnt=contours[i]
#plt.imshow(im)
'''        
'''
plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
'''
#直线拟合
'''
rows,cols = im.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img = cv2.line(im,(cols-1,righty),(0,lefty),(0,255,0),2)
plt.imshow(img)
'''

'''
im1=im.copy()
imgray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
ret1,thresh1 = cv2.threshold(imgray1,127,255,0)
image1, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(0,len(contours)):
    if contours[i].size>500: #轮廓长度大于500个像素点，则记为目标轮廓
        cv2.drawContours(im1,contours,i,(255,255,255),-1)#-1可以填充轮廓内部
        edge=np.mat(contours[i])
'''

#plt.axis("off")#去除坐标轴
