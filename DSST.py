#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
* @Author: Yang Yunhui
* @Date: 2020/7/16
* @Email: yyhbuaa@buaa.edu.cn
* @CopyRight (c) 2020, Yunhui Yang, all rights reserved.
'''

'''
roi: (x,y,w,h)
定位坐标：(x,y)
h,w=image.shape
'''
import sys
import os
import numpy as np

import cv2
from utils import *
import fhog


class DSSTtracker:
    def __init__(self, hog=False, fixed_windows=True, multi_scale=False):
        self.lambdar = 0.0001
        self.pad = 2.5  # 目标扩展出来的区域. w*pad，pad是补零系数
        self.output_sigma_factor = 0.125  # 高斯目标的带宽

        self.multi_scale = multi_scale  ##是否使用尺度估计，即DSST算法
        if self.multi_scale:
            self.template_size = 96  # 模板大小，在计算_tmpl_sz时，较大边长被归一成96，而较小边长按比例缩小
            self.scale_padding = 1.0
            self.scale_step = 1.05  # scale_factor. 抽取目标周围a^nPXa^nR系列尺度的大小
            self.scale_sigma_factor = 0.25  # 这个是不是和高斯带宽有关??
            self.n_scales = 33  # 尺度的数量。

            self.scale_lr = 0.025  # 尺度滤波器学习速率，即论文中的公式（5）
            self.scale_max_area = 512  # 尺度的最大区域。
            self.scale_lambda = 0.01  # 正则系数

            if hog == False:
                print('HOG feature is forced to turn on.')
        elif fixed_windows:  # 不采用多尺度
            self.template_size = 96
            self.scale_step = 1  # scale_factor为1，目标尺度不发生变化
        else:
            self.template_size = 1
            self.scale_step = 1

        if hog or multi_scale:  # 在多尺度中默认采用hog特征
            self.hogFeature = True
        else:
            self.hogFeature = False

        if self.hogFeature:  # 启用hog
            self.interp_factor = 0.012  # 自适应插值因子. 作用：参数更新时决定新计算结果和历史计算结果相互占据的比例，主要在self.train中使用
            self.sigma = 0.6  # 高斯卷积核带宽.sigma越大数据越分散，sigma越小数据越集中
            self.cell_size = 4  # HOG元胞数组尺寸为4个像素. int(self._tmpl_sz[0])//(2*self.cell_size)
            print('hogFeature initializing, wait for a while.')
        else:  # 灰度图像，单通道
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1

        self._tmpl_sz = [0, 0]  # 滤波模板大小
        self._roi = [0., 0., 0., 0.]  # (x,y,w,h)
        self.size_patch = [0., 0., 0.]  # (w,h,c)  //size_patch用于保存截取下来的ROI区域的h，w，c用处：汉宁窗。
        self._scale = 1  # 将最大的边缩小到96，_scale是缩小比例

        self._alphaf = None  # (size_patch[0], size_patch[1], 2) _alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        # alphaf就是滤波器训练后的参数，原文公式Eq 16
        self._prob = None  # (size_patch[0], size_patch[1], 2) _prob是初始化时的高斯响应图
        # _prob就是原文公式 17 中的y帽子
        self._tmpl = None  # raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1])
        # _tmpl是截取的特征的加权平均。 本质就是 hann*featureMaps, 然后和之前的_tmpl做加权平均
        # 主要作用：存储上一个时刻或帧的信息。self._tmpl存储上一帧的结果，self.getFeatures(image)可以获取当前帧的结果。
        # 初始化的时候self._tmpl也是由self.getFeatures(image,initHann=True)获取的
        self.hann = None  # raw: (size_patch[0], size_patch[1]) hog: (size_patch[2], size_patch[0]*size_patch[1]) 存储汉宁窗

        # 尺度性能
        self.currentScaleFactor = 1  # 目标的尺度。1表示和输入的尺度一样，0.5表示目标尺度变成原来的1/2了
        self.base_width = 0  # 初始ROI的宽度
        self.base_height = 0  # 初始ROI的高度
        self.scaleFactors = None  # 所有尺度的变化速率，从大到小，1是中间数

        # scale_model_width = base_width * scale_model_factor
        # scale_model_height = base_height * scale_model_factor
        # scale_max_area = 512
        self.scale_model_width = 0  # the model width for scaling
        self.scale_model_height = 0

        # scale_model_factor 在 min_scale_factor ~ max_scale_factor
        self.min_scale_factor = 0.  # 尺度变化中，最小尺度变化率.
        self.max_scale_factor = 0.

        self.sf_den = None  # ???
        self.sf_num = None

        self.s_hann = None  # 作用：尺度滤波器的汉宁窗
        self.ysf = None  # 作用：保存尺度滤波器的频域Y

        "-------------------------- 位置估计 --------------------------"

    # 三点估计峰值
    def SubPixelPeak(self, left, center, right):
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
        hann2t, hann1t = np.ogrid[0:self.size_patch[0],
                         0:self.size_patch[1]]  # hann2t是纵向一维数组，hann1t是横向一维数组.数值递增，大小分别是从0~size_patch
        hann1t = 0.5 * (
                1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))  # (1-cos)在[0,2pi]的范围内形状类似正态分布函数，两端低中间高
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self.hogFeature:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])  # 变成了一个一维的数组 (100,)
            self.hann = np.zeros((self.size_patch[2], 1),
                                 np.float32) + hann1d  # 加号前半部分生成(n,1)的list。加上hann1d，整体变成了(n,hannd1d.size)的二维数组.
            # 但是(n,2)的list就不能进行类似的操作了
        else:
            self.hann = hann2d

        self.hann = self.hann.astype(np.float32)

    # 高斯函数
    # 仅在第一帧使用，产生高斯响应
    def CreateGaussianPeak(self, sizey, sizex):
        cy, cx = sizey / 2.0, sizex / 2.0
        output_sigma = np.sqrt(sizex * sizey) / self.pad * self.output_sigma_factor  # ?????

        # 因为系数对于任何位置而言都是 1/sqrt(2pi*sigma*sigma)，故在此忽略
        mult = -0.5 / output_sigma / output_sigma
        y, x = np.ogrid[0:sizey, 0:sizex]  # 数值递增大小分别是0~size
        y, x = (y - cy) ** 2, (x - cx) ** 2  # 减去中间数并取平方，本质上是计算距离.
        res = np.exp(mult * (y + x))  # (y+x)得到是(sizey,sizex)大小的数组，每一行都是x的各个元素和y[i]相加。数值本质是(y-cy)**2+(x-cx)**2
        # res所以也是一个(sizey,sizex)大小的数组，每个位置都是计算得到的高斯概率值y
        return fftd(res)  # 返回傅里叶变换之后的Y.论文原文就是y帽子

    # 高斯相关性
    # 输入X和Y的大小必须相同都是 MXN。二者必须是周期的，通过一个cos窗口进行预处理（汉宁窗??）
    def GaussianCorrelation(self, x1, x2):
        if self.hogFeature:
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)  # 存储多个通道的卷积结果之和，原论文公式（31）
            for i in range(self.size_patch[2]):  # 遍历channel通道
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                # 从汉宁窗的生成我们也可以看出数据从(c,h,w)??->(c,w*h)，这个辅助参数就是恢复原来的形状
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                # 本质上是卷积之后x1和x2在fft之后卷积，conjB表示对第二个输入进行共轭
                ##########?????????????问题：公式31交换共轭的位置会不会影响计算结果。原则上输入x1和x2在地位上是等价的，但是为什么公式上采用第一个共轭呢，实际有没有影响
                caux = Real(fftd(caux, True))  # backwards=True。傅里叶逆变换，并选取实部
                c += caux  # 各个通道计算结果相加
            c = rearange(c)  # rearrange是自定义的函数，用于将FFT输出中的直流分量移动到频谱的中央。
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
            c = fftd(c, True)
            c = Real(c)
            c = rearange(c)

        # 加上 x1和x2的范数
        if x1.ndim == 3 and x2.ndim == 3:
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                    self.size_patch[0] * self.size_patch[1] * self.size_patch[2])  # 公式31 ???不过为什么除以维度总数--回答：模量求值公式
            # size_patch[2]序号没有3
        elif x1.ndim == 2 and x2.ndim == 2:
            d = (np.sum(x1 * x1 + x2 * x2) - 2 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        d = d * (d >= 0)  # d小于0的时候，d为0
        d = np.exp(-d / (self.sigma * self.sigma))  # 公式31
        return d

    # 初始化
    # 使用第一帧和他的跟踪框，初始化KCF跟踪器
    def init(self, roi, image):  # roi=(x,y,w,h)
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)

        # _tmpl是截取的特征的加权
        self._tmpl = self.getFeatures(image, inithann=True)  # 提取特征.True表示初始化hanning窗
        # _prob是初始化时的高斯响应图
        self._prob = self.CreateGaussianPeak(self.size_patch[0],
                                             self.size_patch[1])  # 高斯分布 !!!注意输入size_patch[0]，size_patch[1]
        # _alphadf是频域中的相关滤波模板，有两个通道分别是实部和虚部
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)  #

        if self.multi_scale:
            self.DsstInit(self._roi, image)  # 这里是初始化尺度滤波器

        self.train(self._tmpl, train_interp_factor=1.0)  # 初始化训练的时候更新的系数为1.0，表明目前输入占参数变换的100%

    # 从图像上获取ROI区域，提取目标区域的featureMaps。
    # 提取hog特征并与汉宁窗口相乘
    def getFeatures(self, image, inithann=True, scale_adjust=1.):
        cx = self._roi[0] + self._roi[2] / 2  # x+w/2
        cy = self._roi[1] + self._roi[3] / 2  # y+h/2

        if inithann:
            padded_w = self._roi[2] * self.pad  # w*pad
            padded_h = self._roi[3] * self.pad

            if self.template_size > 1:  # 模板大小大于1
                # 将最大的边缩小到96，_scale是缩小比例
                # _tmpl_sz是滤波模板的大小，也是原图上剪裁后的Patch大小
                if padded_w >= padded_h:  # 补零后，padded_w是最大边，缩小为96
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)  # 这里可以看出ROI给出_tmpl_sz的比例，template_size限制了_tmpl_sz的大小
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:  # template_size模板大小等于1
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            if self.hogFeature:
                # 调整_tmpl_sz的尺寸
                # 调整为能够刚好完全覆盖_tmpl_sz的2*self.cell_size的最小整数倍尺寸
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (
                        2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (
                        2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2  # 奇数，会少1
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2  # 奇数，会少1

        # 确定从原图上扣取图片的大小。再根据(cx,cy)调整roi前两个参数的(x,y)
        # 将最大的边缩小到96，_scale是缩小比例
        # 关键部分是self._scale*self._tmpl_sz[0] 将调整后的_tmpl_sz恢复成略大于ROI的区域
        # 这样处理后可以让ROI按照template_size缩小后，满足特定的尺寸要求。
        extracted_roi = [0., 0., 0., 0.]
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0] * self.currentScaleFactor)
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1] * self.currentScaleFactor)

        extracted_roi[0] = int(cx - extracted_roi[2] / 2)  # 能不能改写成 cx - extracted_roi[2] // 2
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        # z是当前被裁剪下来的搜索区域
        z = subWindow(image, extracted_roi, cv2.BORDER_REPLICATE)  # cv2.BORDER_REPLICATE边界复制
        if z.shape[0] != self._tmpl_sz[1] or z.shape[1] != self._tmpl_sz[0]:
            z = cv2.resize(z, tuple(self._tmpl_sz))  # 将z缩小到_tmpl_sz的大小

        # 这一步改变了size_patch数值
        if self.hogFeature:
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0,
                    'map': 0}  # sizeX和sizeY是特征图的宽和高，应该和高斯函数的两个参数对应,numFeatures对应通道数。 ???map是啥
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)  # 调用hog(image, k, mapp)提取特征。
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)  # 归一化并且截断
            mapp = fhog.PCAFeatureMaps(mapp)

            # size_patch用于保存截取下来的特征图的 h，w，c
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp[
                'numFeatures']]))  # map函数用来进行int操作 !!!这里的顺序mapp['sizeY'], mapp['sizeX']
            featuresMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1],
                                               self.size_patch[2])).T  ## (size_patch[2], size_patch[0]*size_patch[1])

        else:  # 不采用HOG特征，就将图片转化为单通道图片
            if z.ndim == 3 and z.shape[2] == 3:
                featuresMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            elif z.ndim == 2:
                featuresMap = z

            # 对灰度图像采用归一化处理，数值分布为-0.5~0.5
            featuresMap = featuresMap.astype(np.float32) / 255.0 - 0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]  ###???z的h和w，是不是[z.shape[1], z.shape[0], 1]

        if inithann:
            self.CreateHanningMats()

        featuresMap = self.hann * featuresMap  # 添加汉宁窗减少频谱泄露
        return featuresMap  ##self._tmpl

    # 根据初始化的图片和框，训练滤波器
    # x就是给出区域的图片的featuresMap，train_interp_factor是interp_factor
    def train(self, x, train_interp_factor=1.0):
        k = self.GaussianCorrelation(x, x)
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)  # 原文公式 17

        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x  # train_interp_factor=1的话就是输入x
        # _alphaf是频域中相关滤波模板(滤波器参数)的加权平均
        # train_interp_factor=1的话就是alphaf
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    # 检测当前帧的目标
    # z是前一帧的训练/第一帧的初始化结果，x是当前帧当前尺度下的特征，peak_value是检测结果峰值
    def detect(self, z, x):  # z是已知的结果，x是当前输入
        k = self.GaussianCorrelation(x, z)
        # 获取响应图
        res = Real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))  # 频域相乘之后傅里叶逆变换

        # pv:响应最大值 pi:最大响应点的位置
        _, pv, _, pi = cv2.minMaxLoc(res)
        p = [float(pi[0]), float(pi[1])]

        # 三点估计峰值位置
        if pi[0] > 0 and pi[0] < res.shape[1] - 1:  # pi不在边界上
            left = res[pi[1], pi[0] - 1]
            right = res[pi[1], pi[0] + 1]
            p[0] += self.SubPixelPeak(left, pv, right)
        if pi[1] > 0 and pi[1] < res.shape[0] - 1:  # pi不在边界上
            left = res[pi[1] - 1, pi[0]]
            right = res[pi[1] + 1, pi[0]]
            p[1] += self.SubPixelPeak(left, pv, right)

        # 再结合res的尺寸，可以得到roi坐标(x,y,w,h)
        p[0] -= res.shape[1] / 2.0  # x--w.img.shape返回的是h和w，第一个维度是h，第二个维度才是w
        p[1] -= res.shape[0] / 2.0  # y--h

        # 返回定位坐标和峰值
        return p, pv

    # 输入当前帧，更新定位坐标
    def update(self, image):  # image是当前帧的输入
        # 修正边界。观察roi的定位坐标(x,y)有没有超出边界. 目标roi的框应当与图片有重合部分，是不是全部在原图内没有关系
        if self._roi[0] + self._roi[2] <= 0: self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0: self._roi[1] = -self._roi[3] + 1
        if self._roi[0] >= image.shape[1] - 1: self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1: self._roi[1] = image.shape[0] - 2

        # 尺度不发生变化的时候，检测峰值及其位置
        # self._tmpl是上一个时刻的计算结果
        loc, peak_value = self.detect(self._tmpl,
                                      self.getFeatures(image=image, inithann=0, scale_adjust=1.0))  # !!!这里汉宁窗的初始化需要设置为0
        # self._tmpl在hog的条件下是特征表达，在~hog的时候是灰度图像输入

        # 原来跟踪框的中心
        cx = self._roi[0] + self._roi[2] / 2.0
        cy = self._roi[1] + self._roi[3] / 2.0
        # 根据loc的信息调整目标框。
        # loc的大小是在roi(x,y,w,h)的基础上进行识别的，所以(x,y)就是loc坐标的原点(0,0),所以loc的本质是关于定位坐标(x,y)的偏移量
        self._roi[0] = self._roi[0] + (loc[0] * self._scale * self.currentScaleFactor) * self.cell_size
        self._roi[1] = self._roi[1] + (loc[1] * self._scale * self.currentScaleFactor) * self.cell_size
        # ?????这里为什么要乘以 cell_size. 估计和hog特征有关，如果没有hog，这个数值是1

        "--------------------------- Scale estimation ---------------------------"
        if self.multi_scale:
            # 修正边界。观察roi的定位坐标(x,y)有没有超出边界. 目标roi的框应当与图片有重合部分，是不是全部在原图内没有关系
            if self._roi[0] + self._roi[2] <= 0: self._roi[0] = -self._roi[2] + 2
            if self._roi[1] + self._roi[3] <= 0: self._roi[1] = -self._roi[3] + 2
            if self._roi[0] >= image.shape[1] - 1: self._roi[0] = image.shape[1] - 1
            if self._roi[1] >= image.shape[0] - 1: self._roi[1] = image.shape[0] - 1  ##相比开始调整了些数值

            # 更新尺度
            scale_pi = self.detect_scale(image)  # 检测当前输入帧的图像尺度
            self.currentScaleFactor = self.currentScaleFactor * self.scaleFactors[scale_pi[0]]
            # scaleFactors所有尺度的变化速率，从大到小，1是中间数
            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            # 到这里为止仅仅是尺度系数currentScaleFactor发生了变化, roi还没有变化

            self.train_scale(image)  # 训练尺度估计器.最后会update_roi，进而改变roi

        # 修正边界。观察roi的定位坐标(x,y)有没有超出边界. 目标roi的框应当与图片有重合部分，是不是全部在原图内没有关系
        if self._roi[0] + self._roi[2] <= 0: self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0: self._roi[1] = -self._roi[3] + 2
        if self._roi[0] >= image.shape[1] - 1: self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1: self._roi[1] = image.shape[0] - 1  ##相比开始调整了些数值
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        # 使用当前的检测框来训练跟踪器，去追踪下一帧的位置。但是不初始化汉宁窗
        x = self.getFeatures(image=image, inithann=0, scale_adjust=1.0)
        ##???不过为什么不用初始化汉宁窗，getFeatures会在提取特征的时候会改变 self.size_patch.

        self.train(x, self.interp_factor)  # self.interp_factor表示参数更新时启用上一次的历史参数值
        # train应该不改动roi的尺寸

        return self._roi  # (x,y,w,h)

    #################
    ### 尺度估计器 ###
    #################

    # 计算y的频域结果Y。这个y用于scale滤波器的，故用Ysf标记
    def ComputeYsf(self):
        # self.n_scales = 33  # default: 33，尺度估计器样本数
        scale_sigma2 = (self.n_scales / self.n_scales ** 0.5 * self.scale_sigma_factor) ** 2  # scale_sigma2指得是平方
        # self.n_scales / self.n_scales**0.5不是等于self.n_scales**0.5

        _, res = np.ogrid[0:0, 0:self.n_scales]  # res是行向量数值是0~n_scales-1。格式是[[0~n_scales-1]]
        # 这一个操作和CreateGaussianPeak很像

        ceilS = np.ceil(self.n_scales / 2.0)  # np.ceil是向上取整。33/2向上取整是17
        res = np.exp(-0.5 * ((res + 1 - ceilS) * (res + 1 - ceilS)) / scale_sigma2)  # (res + 1 - ceilS)是一中心对称的数组
        # 上式的本质是 e^(-1/(2sigma2)*(x-niu)^2) niu正好是平均值17
        # res本质是以关于数组中心对称的高斯函数

        return fftd(res)

    def CreateHanningMatsForScale(self):
        _, hann_s = np.ogrid[0:0, 0:self.n_scales]  # hann_s的数据结构是 [[0~n_scales]]
        hann_s = 0.5 * (1 - np.cos(2 * np.pi * hann_s / (self.n_scales - 1)))
        return hann_s

    # 初始化尺度滤波器（尺度估计器）
    def DsstInit(self, roi, image):
        self.base_width = roi[2]
        self.base_height = roi[3]

        # 计算尺度高斯峰值（输出结果是在频域，是Y）
        self.ysf = self.ComputeYsf()
        self.s_hann = self.CreateHanningMatsForScale()  # 初始化汉宁窗

        # 获取所有的尺度变化速率
        scaleFactors = np.arange(self.n_scales)  # range是返回一个类，np.arange是返回一个array
        ceilS = np.ceil(self.n_scales / 2.0)
        self.scaleFactors = np.power(self.scale_step, ceilS - scaleFactors - 1)
        # self.scale_step就是尺度变化的速度a=1.02，(ceilS - scaleFactors - 1) = (17-[1~33]) = [-(S-1)/2,...,(S-1)/2] =n 的范围
        # self.scaleFactors = a^nPXa^nR

        # 命名规律 scale_model 是尺度模型的意思。是一个整体的前缀
        # 获取尺度速率(scaling rate) 来压缩模型尺寸??
        scale_model_factor = 1.  # 这是一个类似_scale的参数，表征将图片压缩的比例。
        if self.base_height * self.base_width > self.scale_max_area:
            scale_model_factor = (self.scale_max_area / self.base_height / self.base_width) ** 0.5
            # 面积之比的开方，表征单个维度的缩小比例

        self.scale_model_height = int(self.base_height * scale_model_factor)
        self.scale_model_width = int(self.base_width * scale_model_factor)

        # 计算最小和最大的尺度rate
        # scale_model_factor 在 min_scale_factor ~ max_scale_factor
        self.min_scale_factor = np.power(self.scale_step, np.ceil(
            np.log((max(5 / self.base_width, 5 / self.base_height) * (1 + self.scale_padding))) / 0.0086))
        # 最小尺度是5*5
        self.max_scale_factor = np.power(self.scale_step, np.floor(np.log((min(image.shape[0] / self.base_width,
                                                                               image.shape[1] / self.base_height) * (
                                                                                   1 + self.scale_padding))) / 0.0086))
        # 最大尺度是image.shape[0]*image.shape[1]

        self.train_scale(image, True)

    # 获取尺度样本。提取hog特征并与汉宁窗口相乘。
    # 和getFeatures函数相似
    def get_scale_sample(self, image):
        xsf = None  # x是输入，s是scale，f是傅里叶变换
        for i in range(self.n_scales):
            # subwindow的尺寸，等待被检测
            patch_width = self.base_width * self.scaleFactors[i] * self.currentScaleFactor
            patch_height = self.base_height * self.scaleFactors[i] * self.currentScaleFactor

            cx = self._roi[0] + self._roi[2] / 2
            cy = self._roi[1] + self._roi[3] / 2

            # 获得subwindow，扣取图片
            im_patch = extractImage(image, cx, cy, patch_width,
                                    patch_height)  # im_patch是提取的区域。extractImage是类似subwindow, 但是subwindows是在频域的
            if self.scale_model_width > im_patch.shape[1]:
                # 如果scale_model_width大于截取图片的尺寸。scale_model_width表示将图片保持在template以内尺寸。
                # 如果im_patch小于scale_model_width，说明要放大，上采样
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0,
                                              1)  # 最后的插值方式不同
            else:  # 下采样
                im_patch_resized = cv2.resize(im_patch, (self.scale_model_width, self.scale_model_height), None, 0, 0,
                                              3)

            # 提取hog特征
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            # sizeX和sizeY是特征图的宽和高，应该和高斯函数的两个参数对应,numFeatures对应通道数。 ???map是啥
            mapp = fhog.getFeatureMaps(im_patch_resized, self.cell_size, mapp)  # 调用hog(image, k, mapp)提取特征。
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)  # 归一化并且截断
            mapp = fhog.PCAFeatureMaps(mapp)

            if i == 0:
                totalSize = mapp['numFeatures'] * mapp['sizeX'] * mapp['sizeY']
                xsf = np.zeros((totalSize, self.n_scales))

            # 将FHOG的结果和汉宁窗相乘
            featuresMap = mapp['map'].reshape((totalSize, 1))
            # hann_s的数据结构是 [[0~(totalSize, 1)]]
            featuresMap = self.s_hann[0][i] * featuresMap  # （1*totalSize）
            xsf[:, i] = featuresMap[:, 0]  # 这样复制等价于深拷贝(已验证)

        return fftd(xsf, backWards=False, byRow=True)

    # 训练尺度估计器
    def train_scale(self, image, init=False):
        xsf = self.get_scale_sample(image)  # 获取n_scales个特征featuresMap。xsf(totalsize,n_scales)

        # Adjust ysf to the same size as xsf in the first time
        # 在第一帧的时候调整ysf到xsf相同的形状。ysf的尺寸是array [-xx~xx] 尺寸是1*n_scales
        if init:
            totalSize = xsf.shape[0]
            self.ysf = cv2.repeat(self.ysf, totalSize, 1)  # ysf在第一个维度重复totalSize次. 在自己的维度就是自己。(1代表重复次数)

        # Get new GF in the paper (delta A)
        # num指的是numerator--分子，DSST论文中的At
        new_sf_num = cv2.mulSpectrums(self.ysf, xsf, 0, conjB=True)

        # den指得是denominator--分母，DSST论文中的Bt
        new_sf_den = cv2.mulSpectrums(xsf, xsf, 0, conjB=True)
        new_sf_den = cv2.reduce(Real(new_sf_den), 0, cv2.REDUCE_SUM)  # 0表示数据被处理成一行(1,n_scales).输出结果是每一列之和

        # 更新分子分母
        if init:
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            # Get new A and new B
            self.sf_den = cv2.addWeighted(self.sf_den, (1 - self.scale_lr), new_sf_den, self.scale_lr, 0)
            # 历史参数和现在计算结果加权。
            # self.sf_den = self.sf_den*(1 - self.scale_lr)+new_sf_den*self.scale_lr
            self.sf_num = cv2.addWeighted(self.sf_num, (1 - self.scale_lr), new_sf_num, self.scale_lr, 0)

        self.update_roi()

    def update_roi(self):
        cx = self._roi[0] + self._roi[2] / 2
        cy = self._roi[1] + self._roi[3] / 2

        # 重新计算框的长宽
        self._roi[2] = self.base_width * self.currentScaleFactor
        self._roi[3] = self.base_height * self.currentScaleFactor

        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

    # 检测当前图像尺度
    def detect_scale(self, image):
        xsf = self.get_scale_sample(image)  # xsf尺寸(totalSize, self.n_scales)

        # Compute AZ in the paper
        add_temp = cv2.reduce(complexMultiplication(self.sf_num, xsf), 0, cv2.REDUCE_SUM)  # DSST论文公式6

        # compute the final y
        # complexDivisionReal 是虚数除以实数
        scale_response = cv2.idft(complexDivisionReal(add_temp, (self.sf_den + self.scale_lambda)), None,
                                  cv2.DFT_REAL_OUTPUT)  # cv2.idft是傅里叶反变换

        # Get the max point as the final scaling rate
        # pv:响应最大值 pi:相应最大点的索引数组
        _, pv, _, pi = cv2.minMaxLoc(scale_response)

        return pi  # 返回最大的位置


if __name__ == "__main__":
    # aa, bb = np.ogrid[0:10, 0:10]
    # aa = 0.5 * (1 - np.cos(2 * np.pi * aa / (10 - 1)))
    # bb = 0.5 * (1 - np.cos(2 * np.pi * bb / (10 - 1)))
    # tmp = aa * bb
    # # print(tmp,tmp.shape)
    # tmp = tmp.reshape(10 * 10)

    # aa = [i for i in range(10)]
    # bb = [i ** 2 for i in range(10)]
    # aa[:] = bb[:]
    # print(aa)
    # bb[0] = 100
    # print(aa)

    # hann = np.zeros((3, 2), np.float32) + tmp
    # # hann2=np.zeros((3, 1), np.float32)
    # print(hann, hann.shape)

    # from tqdm import tqdm
    # for i in tqdm(range(1000)):
    #     i+=1
    #     pass

    aa, bb = np.ogrid[0:10, 0:10]
    yy = cv2.repeat(bb, 10, 2)
    print(yy)
