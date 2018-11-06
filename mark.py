#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:43:34 2018

@author: sanglinwei
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#read data 
data_1=open(r'D:\big_data\马塘风电场包齿齿轮箱数据\包齿数据\A1-007.csv')
data=pd.read_csv(data_1)
wind_1=data[['风速','功率']]
wind=wind_1[wind_1['功率']>0]
wind2=wind_1[wind_1['功率']<=0]
#新建wind的索引
wind.index=range(0,wind.shape[0])
wind_speed=wind['风速']
wind_power=wind['功率']

#mark picture Pic
im_mark=cv2.imread(r'D:/big_data/demo_1.png')
#画图
fig=plt.figure()
plt.rcParams['figure.figsize'] = (6.0, 4.0)#比例
plt.rcParams['savefig.dpi'] = 72 #图片像素
plt.rcParams['figure.dpi'] = 72 #分辨率
#plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
ax=fig.add_subplot(1,1,1)
ax.scatter(wind_speed,wind_power,c="k",alpha=1,s=1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

k1=wind['风速'].argmax()
k2=wind['风速'].argmin()
k3=wind['功率'].argmax()
k4=wind['功率'].argmin()


ax.scatter(wind_speed[k1],wind_power[k1],c="r",alpha=1,s=1)
plt.savefig(r'D:/big_data/pic_1.png',dpi=72)
im=cv2.imread(r'D:/big_data/pic_1.png')
im3=im.copy()
#find 
for i in range(288):
    for j in range(432):
        if all(im3[i,j]==[0,0,255]):
           a=i;
           b=j;
           break;
x2=432-b;
ax.scatter(wind_speed[k1],wind_power[k1],c="k",alpha=1,s=1)
           
ax.scatter(wind_speed[k2],wind_power[k2],c="r",alpha=1,s=1)
plt.savefig(r'D:/big_data/pic_1.png',dpi=72)
im=cv2.imread(r'D:/big_data/pic_1.png')
im3=im.copy()
#find x1
for i in range(288):
    for j in range(432):
        if all(im3[i,j]==[0,0,255]):
           a=i;
           b=j;
           break;
x1=b-1;
ax.scatter(wind_speed[k2],wind_power[k2],c="k",alpha=1,s=1)           

ax.scatter(wind_speed[k3],wind_power[k3],c="r",alpha=1,s=1)
plt.savefig(r'D:/big_data/pic_1.png',dpi=72)
im=cv2.imread(r'D:/big_data/pic_1.png')
im3=im.copy()
#find y1
for i in range(288):
    for j in range(432):
        if all(im3[i,j]==[0,0,255]):
           a=i;
           b=j;
           break;           
y1=a-1;
ax.scatter(wind_speed[k3],wind_power[k3],c="k",alpha=1,s=1)

ax.scatter(wind_speed[k4],wind_power[k4],c="r",alpha=1,s=1)
plt.savefig(r'D:/big_data/pic_1.png',dpi=72)
im=cv2.imread(r'D:/big_data/pic_1.png')
im3=im.copy()

#find y2
for i in range(288):
    for j in range(432):
        if all(im3[i,j]==[0,0,255]):
           a=i;
           b=j;
           break;
y2=288-a;
ax.scatter(wind_speed[k4],wind_power[k4],c="k",alpha=1,s=1) 

#推导像素点与实际点之间的关系
tx=(432-x1-x2)/(wind.loc[k1]['风速']-wind.loc[k2]['风速'])
ty=(288-y1-y2)/(wind.loc[k3]['功率']-wind.loc[k4]['功率'])
# data['功率']  data['风速']
dx=np.zeros(data.shape[0])
dy=np.zeros(data.shape[0])
ai=np.zeros(data.shape[0])
bj=np.zeros(data.shape[0])
mark=np.zeros(data.shape[0])

#mark the dataset 
for i in range(0,data.shape[0]):
    if data['功率'][i]>0:        
        dx[i]=(data['风速'][i]-wind.loc[k2]['风速'])*tx
        dy[i]=(data['功率'][i]-wind.loc[k4]['功率'])*ty
        ai[i]=288-dy[i]-y2
        bj[i]=x1+dx[i]
        da=int(ai[i]) 
        db=int(bj[i])
        if all(im_mark[da,db]==[0,0,255]):
            mark[i]=1    # type 1 means normal        
        if all(im_mark[da,db]==[255,0,0]):
            mark[i]=2   # type 2 means discrete 
        if all(im_mark[da,db]==[0,127,0]):
            mark[i]=3     #type 3 means limited
    else:
        mark[i]=0

data['mark']=mark



#### test the pic
fig1=plt.figure()
plt.rcParams['figure.figsize'] = (6.0, 4.0)#比例
plt.rcParams['savefig.dpi'] = 72 #图片像素
plt.rcParams['figure.dpi'] = 72 #分辨率
#plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)
ax1=fig1.add_subplot(1,1,1)

c1=data[data['mark']==1];
c2=data[data['mark']==2];
c3=data[data['mark']==3];
c4=data[data['mark']==0];

ax1.scatter(c4['风速'],c4['功率'],c="y",alpha=1,s=36)
ax1.scatter(c1['风速'],c1['功率'],c="b",alpha=1,s=36)
ax1.scatter(c2['风速'],c2['功率'],c="r",alpha=1,s=36)
ax1.scatter(c3['风速'],c3['功率'],c="g",alpha=1,s=36)


data[data['mark']==1].to_csv('D:/big_data/normal.csv',encoding='utf_8_sig')
data[data['mark']==2].to_csv('D:/big_data/discrete.csv',encoding='utf_8_sig')
data[data['mark']==3].to_csv('D:/big_data/limited.csv',encoding='utf_8_sig')
data[data['mark']==0].to_csv('D:/big_data/belowzero.csv',encoding='utf_8_sig')


ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis('off')

'''
ax.scatter(wind_speed[k1],wind_power[k1],c="r",alpha=1,s=36)
ax.scatter(wind_speed[k2],wind_power[k2],c="r",alpha=1,s=36)
ax.scatter(wind_speed[k3],wind_power[k3],c="r",alpha=1,s=36)
ax.scatter(wind_speed[k4],wind_power[k4],c="r",alpha=1,s=36)
'''
'''
    dx=(wind_speed[i]-wind.loc[k2]['风速'])*tx
    dy=(wind_power[i]-wind.loc[k4]['功率'])*ty
    ai=int(288-dy-y2)
    bj=int(x1+dx)
'''  
          
             

