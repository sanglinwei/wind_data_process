#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:27:10 2018

@author: sanglinwei
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# get image
'''
data=pd.read_csv('/Users/sanglinwei/Downloads/big_data/马塘风电场包齿齿轮箱数据/包齿数据/B3-064.csv',encoding='gbk')
data['mark']=0
data.index=range(0,data.shape[0])
wind_1=data[['风速','功率']]
wind=wind_1[wind_1['功率']>0]
wind_speed=wind['风速']
wind_power=wind['功率']
#画图
fig=plt.figure()
plt.rcParams['figure.figsize'] = (6.0, 4.0)#比例
plt.rcParams['savefig.dpi'] = 72 #图片像素
plt.rcParams['figure.dpi'] = 72 #分辨率
plt.rcParams['axes.linewidth']=1
#plt.rcParams['legend.borderpad']=0.4
#plt.rcParams['figure.subplot.wspace']=0
#plt.rcParams['figure.facecolor']='white'
#plt.rcParams['figure.edgecolor']='white'
#plt.rcParams["figure.frameon"]=False
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
ax=fig.add_subplot(1,1,1)
ax.scatter(wind_speed,wind_power,c="k",alpha=1,s=36)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
#plt.imshow()
#plt.imshow(ax)
plt.savefig('/Users/sanglinwei/Downloads/big_data/pic_1.png',dpi=72)
'''
#image process

im=cv2.imread(r'D:/big_data/pic_1.png')

im1=im.copy()
#找到数据点
im2=im.copy()
im3=im.copy()
#find
''' 
for i in range(288):
    for j in range(432):
        if all(im3[i,j]==[0,0,255]):
           a=i;
           b=j;
           break;
'''           
            
#主体轮廓
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
#产生匹配图像
img = cv2.imread(r'D:/big_data/standard_4.png', 0)
image6, contours6, hierarchy6 = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt5=contours6[0]
for wd in range(7,17):
    kernel = np.ones((wd,wd),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    image2, contours2, hierarchy2 = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    con=cv2.arcLength(contours2[0],True)
    num=len(contours2)
    if(num==1):
       #s=cv2.contourArea(contours2[0])    
        cnt1=contours2[0]
       

        similar4=cv2.matchShapes(cnt1,cnt5,3,0.0)
        if(similar4<1):
            break;
    
'''
#主要调整kernel
wd=7
kernel = np.ones((wd,wd),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

image2, contours2, hierarchy2 = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
con=cv2.arcLength(contours2[0],True)
num=len(contours2)

s=cv2.contourArea(contours2[0])

cnt1=contours2[0]



#相似度匹配
#目标图像4
img = cv2.imread(r'D:/big_data/standard_4.png', 0)
image6, contours6, hierarchy6 = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt5=contours6[0]
similar4=cv2.matchShapes(cnt1,cnt5,3,0.0)
'''
mom = cv2.moments(cnt5)
Humoments = cv2.HuMoments(mom)
print(Humoments)
print('相似程度%.4f'%similar4,'kernel',wd)
#print('相似程度%.4f'%similar,'相似程度%.4f'%similar2,'相似程度%.4f'%similar3,'相似程度%.4f'%similar4,'kernel',wd)

print('轮廓数目',num)

print('轮廓边长',con)
'''
print('轮廓面积',s)
print('x',cx,'y',cy)
'''
#开运算
#正常区域
for i in range(288):
    for j in range(432):
        if all(opening[i,j]==[255,255,255]):
            im[i,j]=[0,0,255]
#            
for i in range(288):
    for j in range(432):
        if all(im1[i,j]==[0,0,0]):
            im[i,j]=[255,0,0]         
#            
for i in range(288):
    for j in range(432):
        if all(im[i,j]==[0,0,0]):
            im[i,j]=[0,127,0] 

plt.imshow(im,'gray')
cv2.imwrite(r'D:/big_data/demo_1.png',im)
#cv2.imwrite('/Users/sanglinwei/Downloads/big_data/standard_3.png',opening)
'''
#目标图像1
img = cv2.imread(r'D:/big_data/standard_1.png', 0)
image3, contours3, hierarchy3 = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt2=contours3[0]
similar=cv2.matchShapes(cnt1,cnt2,1,0.0)
#目标图像2
img = cv2.imread(r'/D:/big_data/standard_2.png', 0)
image4, contours4, hierarchy4 = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#cnt3=contours4[0]
#similar2=cv2.matchShapes(cnt1,cnt3,1,0.0)
#目标图像3
img = cv2.imread(r'D:/big_data/standard_3.png', 0)
image5, contours5, hierarchy5 = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt4=contours5[0]
similar3=cv2.matchShapes(cnt1,cnt4,1,0.0)
#print('相似程度%.4f'%similar,'相似程度%.4f'%similar2,'相似程度%.4f'%similar3,'kernel',wd)
'''
'''
#计算质心
M = cv2.moments(cnt)
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
#外接矩形
x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
rect = cv2.minAreaRect(cnt)  # 最小外接矩形
box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
cv2.drawContours(im, [box], 0, (255, 0, 0), 2)
'''