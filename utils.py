#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
* @Author: Yang Yunhui
* @Date: 2020/7/18
* @Email: yyhbuaa@buaa.edu.cn
* @CopyRight (c) 2020, Yunhui Yang, all rights reserved.
'''
import numpy as np
import json
import cv2, copy


def OPCal(bbox1, bbox2):
    '''
    计算重合率
    '''
    xmin1 = bbox1[0][0]
    ymin1 = bbox1[0][1]
    xmax1 = bbox1[1][0]
    ymax1 = bbox1[1][1]

    xmin2 = bbox2[0][0]
    ymin2 = bbox2[0][1]
    xmax2 = bbox2[1][0]
    ymax2 = bbox2[1][1]

    overArea = min(abs(xmax2 - xmin1), abs(xmax1 - xmin2)) * min(abs(ymax2 - ymin1), abs(ymax1 - ymin2))
    totalArea = (max(xmax1, xmax2) - min(xmin1, xmin2)) * (max(ymax1, ymax2) - min(ymin1, ymin2))

    overlappedRate = overArea / totalArea
    return overlappedRate


def Json2Bbox(jsonPath, trackerType='single'):
    jsInfo = ReadJsonSample(jsonPath)
    if trackerType == 'single':
        return jsInfo['shapes'][0]['points']
    else:
        print("##fixing##")


def Roi2Bbox(roi):
    bbox = [[0., 0.], [0., 0.]]
    bbox[0][0] = roi[0]
    bbox[0][1] = roi[1]

    bbox[1][0] = roi[0] + roi[2]
    bbox[1][1] = roi[1] + roi[3]

    return bbox


def MarkedBboxOnImage(imageIn, bbox, label=None, color=(0, 255, 0), gray=True):
    '''
    在输入图片生绘制框，并返回绘制后的图片
    单目标跟踪，bbox= [[x1,y1],[x2,y2]]
    '''
    image = copy.deepcopy(imageIn)
    pt1 = bbox[0]
    pt2 = bbox[1]
    # print("bbox:", bbox)
    if gray:
        color = 255
    else:
        color = color

    if label != None:
        labelOrg = (int(pt1[0]), int(pt1[1]) - 10)
        cv2.putText(img=image,
                    text=label,
                    org=labelOrg,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=color,
                    thickness=2,
                    )

    cv2.rectangle(img=image,
                  pt1=(int(pt1[0]), int(pt1[1])),
                  pt2=(int(pt2[0]), int(pt2[1])),
                  color=color,
                  thickness=2, )

    return image


def ShowImage(image, winName="Test", scale=2.0, isSave=False):
    try:
        imgW, imgH = image.shape
    except:
        imgW, imgH, _ = image.shape

    imgH = int(imgH / scale)
    imgW = int(imgW / scale)
    img = cv2.resize(image, (imgH, imgW), 0, 0)
    if (isSave):
        # cv2.imwrite('configs/results/sample/' + winName + '.jpg', img)
        pass
    cv2.imshow(winName, img)
    cv2.waitKey(0)


"----------------------------- DSST necessary -----------------------------"


def Real(image):  # 输入的是(:,:,2)--(:,:,0)是实部，(:,:,1)是虚部
    return image[:, :, 0]


def Imag(image):
    return image[:, :, 1]


def fftd(image, backWards=False, byRow=False):
    # DFT_INVERSE: 用一维或二维逆变换取代默认的正向变换,
    # DFT_SCALE: 缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用。
    # DFT_COMPLEX_OUTPUT: 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性

    if byRow:  # 对输入矩阵的每“行”(1，n_scales，尺度方向)进行正向或反向的傅里叶变换
        return cv2.dft(np.float32(image), flags=(cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT))
    else:
        if backWards:
            flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE)
            # DFT_SCALE缩放比例标识符，根据数据元素个数平均求出其缩放结果，如有N个元素，则输出结果以1/N缩放输出，常与DFT_INVERSE搭配使用
        else:
            flags = cv2.DFT_COMPLEX_OUTPUT
            # 对一维或二维的实数数组进行正向变换，这样的结果虽然是复数阵列，但拥有复数的共轭对称性（CCS），可以以一个和原数组尺寸大小相同的实数数组进行填充，这是最快的选择也是函数默认的方法
        return cv2.dft(np.float32(image), flags=flags)  # np.float32和np.float不太一样


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
                                                                       image[0:cy, 0:cx]
    # 右上角和左下角
    image_[0:cy, cx:image.shape[1]], image_[cy:image.shape[0], 0:cx] = image[cy:image.shape[0], 0:cx], \
                                                                       image[0:cy, cx:image.shape[1]]
    return image_


# 输入rect(或者roi)(x,y,w,h)计算右边界 x+w
def rightBorder(rect):
    return rect[0] + rect[2]


def bottomBorder(rect):
    return rect[1] + rect[3]


# 限制宽高。
# 输入roi和limit(xl,yl,wl，hl)。
# 输出的roi四个成员描述的框在limit描述的边框内部
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


# 边框界定
# limitBorder是四维度的，cutOutSize是一维度的
def cutOutSize(num, limit):
    if num < 0:
        num = 0
    if num > limit - 1:
        num = limit - 1

    return int(num)


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
    cutWindow = [x for x in window]  # 类似深拷贝。window= (x,y,w,h)
    cutWindow = limitBorder(cutWindow, [0, 0, image.shape[1], image.shape[0]])  # 将cutwin的坐标和尺寸限制在图片内
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)

    border = getBorder(window, cutWindow)  # 本质上是获取window和cutwindow之间差异--边界。cutwin被修剪了
    res = image[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]  # 在图片上截取cutwin部分

    if border != [0., 0., 0., 0.]:  # 即说明window有部分和图像边界重合了
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
        # top,bottom,left,right分别表示在原图四周扩充边缘的大小
        '''
        borderType:
        1. BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh (复制法，复制边缘重复性扩展)
        2. BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
        '''
    return res


'''
以（cx，cy）为中心截取(patch_width, patch_height)大小的图片
cx, cy, patch_width, patch_height 都应该是float类型数据
'''


def extractImage(image, cx, cy, patch_width, patch_height):
    xmin = np.floor(cx) - np.floor(patch_width / 2)  # floor是向下取整。ceil是向上取整
    xmax = np.floor(cx + patch_width - 1) - np.floor(patch_width / 2)
    xmin, xmax = cutOutSize(xmin, image.shape[1]), cutOutSize(xmax, image.shape[1])

    ymin = np.floor(cy) - np.floor(patch_height / 2)
    ymax = np.floor(cy + patch_height - 1) - np.floor(patch_height / 2)
    ymin, ymax = cutOutSize(ymin, image.shape[0]), cutOutSize(ymax, image.shape[0])

    return image[ymin:ymax, xmin:xmax]
