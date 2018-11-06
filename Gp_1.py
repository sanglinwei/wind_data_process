#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:29:28 2018

@author: sanglinwei
"""
import pandas as pd
import matplotlib.pyplot as plt
data_1=open(r'D:\big_data\马塘风电场包齿齿轮箱数据\包齿数据\A1-007.csv')
data=pd.read_csv(data_1)
#data=pd.read_csv('D:\big_data\数据分析合作-2014年9月9日数据\02-机组功率曲线随时间变化\盱眙-十分钟数据\盱眙56.csv',encoding='gbk')

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

#plt.imshow(ax)
plt.savefig(r'D:\big_data\pic_1.png',dpi=72)


'''
wind2=wind_1[wind_1['功率']<0]
wind2_speed=wind2['风速']
wind2_power=wind2['功率']
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(wind2_speed,wind2_power,c="k",alpha=1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
plt.savefig('/Users/sanglinwei/Downloads/big_data/pic_2.png')
'''