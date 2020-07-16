#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
* @Author: CETC58(CKS) Wuhan Branch Company algorithm group @Yang Yunhui
* @Date: 2020/7/16
* @Email: yyhbuaa@buaa.edu.cn
* @CopyRight (c) 2020, Yunhui Yang, all rights reserved.
'''
import sys
import os
import numpy as np


class DSSTtracker:
    def __init__(self, hog=False, fixed_windows=True, multi_scale=False):
        self.lambdar = 0.0001
        self.pad = 2.5  # 目标扩展出来的区域
        self.output_sigma_factor = 0.125  # 高斯目标的带宽

        self.multi_scale = multi_scale
        if self.multi_scale:
            self.template_size = 96  # 模板大小，在计算_tmpl_sz时，较大边长被归一成96，而较小边长按比例缩小
            self.scale_padding = 1.0
            self.scale_factor = 1.02  # scale_step. 抽取目标周围a^nPXa^nR系列尺度的大小
            self.scale_sigma_factor = 0.25  # 这个是不是和高斯带宽有关??
            self.n_scales = 33  # 尺度的数量。

            self.scale_lr = 0.025  # 尺度滤波器学习速率，即论文中的公式（5）
            self.scale_max_area = 512
            self.scale_lambda = 0.01  # 正则系数

            if ~hog:
                print('HOG feature is forced to turn on.')
        elif fixed_windows:  # 不采用多尺度
            self.template_size = 96
            self.scale_factor = 1  # scale_factor为1，目标尺度不发生变化
        else:
            self.template_size = 1
            self.scale_factor = 1

        if hog or multi_scale:  # 在多尺度中默认采用hog特征
            self.hogFeature = True
        else:
            self.hogFeature = False

        if self.hogFeature:  # 启用hog
            self.interp_factor = 0.012  # 自适应插值因子
            self.sigma = 0.6  # 高斯卷积核带宽.sigma越大数据越分散，sigma越小数据越集中
            self.cell_size = 4  # HOG元胞数组尺寸为4个像素
            print('hogFeature initializing, wait for a while.')
        else:  # 灰度图像，单通道
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1

        self._tmpl_sz = [0, 0]  # template_size模板大小
        self._roi = [0., 0., 0., 0.]  # (x,y,w,h)
        self.size_patch = [0, 0, 0]  # (w,h,c)  //用处：汉宁窗
        self._scale = 1

        self._alpha = None  # (size_patch[0], size_patch[1], 2)
        self._prob = None  # (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1])

        # 尺度性能
        self.currentScaleFactor = 1
        self.base_width = 0  # 初始ROI的宽度
        self.base_height = 0  # 初始ROI的高度
        self.scaleFactors = None  # 所有尺度的变化速率，从大到小，1是中间数
        self.scale_model_width = 0  # the model width for scaling
        self.scale_model_height = 0
        self.min_scale_factor = 0.  # 尺度变化中，最小尺度变化率
        self.max_scale_factor = 0.

        self.sf_den = None  # ???
        self.sf_num = None

        self.s_hann = None
        self.ysf = None

        "-------------------------- 位置估计 --------------------------"

    # 三点估计峰值
    def subPixelPeak(self, left, center, right):
        '''
        y=ax^2+bx+c
        a=0.5(left+right)-center
        b=0.5(right-left)
        c=center

        x_=-b/(2a)=0.5 * (right - left) / ((left+right)-2center)
        '''
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)  # 返回最大位置

    # 初始话hanning窗口，仅在第一帧使用
    # 目的是采样时为不同的样本分配不同的权重
    def CreateHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]  # hann2t是纵向一维数组，hann1t是横向一维数组
        hann1t=0.5*(1-np.cos(2*np.pi*hann1t/(self.size_patch[1]-1)))
        hann2t=0.5*(1-np.cos(2*np.pi*hann2t/(self.size_patch[1]-1)))
        hann2d=hann2t*hann1t
        