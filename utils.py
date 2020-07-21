#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
* @Author: CETC58(CKS) Wuhan Branch Company algorithm group @Yang Yunhui
* @Date: 2020/7/18
* @Email: yyhbuaa@buaa.edu.cn
* @CopyRight (c) 2020, Yunhui Yang, all rights reserved.
'''
import numpy as np


def Real(image):  # 输入的是(:,:,2)--(:,:,0)是实部，(:,:,1)是虚部
    return image[:, :, 0]


def Imag(image):
    return image[:, :, 1]


def fftd(image, backWards=False, byRow=False):
    # DFT_INVERSE: 用一维或二维逆变换取代默认的正向变换,
    # DFT_SCALE: 缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用。
    # DFT_COMPLEX_OUTPUT: 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性

    if byRow:  # 对输入矩阵的每“行”(1，n_scales，尺度方向)进行正向或反向的傅里叶变换
        return cv2.dft(np.float(image), flags=(cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT))
    else:
        if backWards:
            flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE)
            # DFT_SCALE缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用
        else:
            flags = cv2.DFT_COMPLEX_OUTPUT
            # 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性（CCS），可以以一个和原数组尺寸大小相同的实数数组进行填充，这是最快的选择也是函数默认的方法
        return cv2.dft(np.float(image), flags=flags)


# 两个复数，它们的积 (a+bi)(c+di)=(ac-bd)+(ad+bc)i
def complexMultiplication(a, b):
    res = np.zeros(a.shape, dtype=a.dtype)
    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


# 两个复数，它们相除 (a+bi)/(c+di)=(ac+bd)/(c*c+d*d) +((bc-ad)/(c*c+d*d))i
def complexDivision(a, b):
    res = np.zeros(a.shape, dtype=a.dtype)
    mul = 1.0 / (b[:, :, 0] * b[:, :, 0] + b[:, :, 1] * b[:, :, 1])
    res[:, :, 0] = mul * (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1])
    res[:, :, 1] = mul * (a[:, :, 1] * b[:, :, 0] - a[:, :, 0] * b[:, :, 1])
    return res


# 虚数除以实数
def complexDivisionReal(a, b):
    res = np.zeros(a.shape, dtype=a.dtype)
    mul = 1. / b
    res[:, :, 0] = mul * (a[:, :, 0])
    res[:, :, 1] = mul * (a[:, :, 1])
    return res


# 将FFT输出的直流分量移动到频谱中央
def rearange(image):
    assert (image.ndim == 2)  # ndim为数组维度
    image_ = np.zeros(image.shape, dtype=image.dtype)
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    # 左上角和右下角互换
    image_[0:cy, 0:cx], image_[cy:image.shape[0], cx:image.shape[1]] = image[cy:image.shape[0], cx:image.shape[1]], \
                                                                       image_[0:cy, 0:cx]
    # 右上角和左下角
    image_[0:cy, cx:image.shape[1]], image_[cy:image.shape[0], 0:cx] = image_[cy:image.shape[0], 0:cx], \
                                                                       image_[0:cy, cx:image.shape[1]]
    return image_


# 输入rect(或者roi)(x,y,w,h)计算右边界 x+w
def rightBorder(rect):
    return rect[0] + rect[2]


def bottomBorder(rect):
    return rect[1] + rect[3]


# 限制宽高。
# 输入roi和limit(xl,yl,wl，hl)
def limitBorder(rect, limits):
    if rect[0] + rect[2] > limits[0] + limits[2]:
        rect[2] = limits[0] + limits[2] - rect[0]
    if rect[0] < limits[0]:
        rect[2] = rect[2] + rect[0] - limits[0]
        rect[0] = limits[0]
    if rect[2] < 0:
        rect[2] = 0

    if rect[1] + rect[3] > limits[1] + limits[3]:
        rect[3] = limits[1] + limits[3] - rect[1]
    if rect[1] < limits[1]:
        rect[3] = rect[3] + rect[1] - limits[1]
        rect[1] = limits[1]
    if rect[3] < 0:
        rect[3] = 0

    return rect


# 取超出来的边界
# ??感觉就像是original包裹住了limited
def getBorder(original, limited):
    res = [0., 0., 0., 0.]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = rightBorder(original) - rightBorder(limited)
    res[3] = bottomBorder(original) - bottomBorder(limited)
    assert (np.all(np.array(res) >= 0))  # 以上计算结果必须大于0
    return res


# 空间域和频域滤波处理之前需要考率图像的边界情况
# 处理方法是为图像增加一定的边缘，适应卷积核在原图像边缘的操作
# 用例： subWindow(image, extracted_roi, cv2.BORDER_REPLICATE)
def subWindow(image, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow=[x for x in window] #类似深拷贝。window= (x,y,w,h)
    cutWindow=limitBorder(cutWindow,[0,0,image.shape[1],image.shape[0]])
    assert (cutWindow[2]>0 and cutWindow[3]>0)
    return


def rearrange():
    return


def extractImage(image, cx, cy, patch_width, patch_height):
    return
