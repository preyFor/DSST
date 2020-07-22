#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
* @Author: Yang Yunhui
* @Date: 2020/7/21
* @Email: yyhbuaa@buaa.edu.cn
* @CopyRight (c) 2020, Yunhui Yang, all rights reserved.
'''
from DSST import *
from utils import *

import os
import json
import sys


def jsPathReader(dataBasePath):
    js_lst = []
    for file in os.listdir(dataBasePath):
        if file[-4:] == 'json':
            js_path = dataBasePath + '/' + file
            js_lst.append(js_path)
    return js_lst


def jsReader(jsPath):
    fp = open(jsPath, mode='rb')
    js_info = json.load(fp)
    return js_info


def points2Roi(points):
    roi = [0., 0., 0., 0.]
    roi[0] = points[0][0]  # x
    roi[1] = points[0][1]  # y

    roi[2] = points[1][0] - points[0][0]  # w
    roi[3] = points[1][1] - points[0][1]  # h

    return roi


def test01():
    dataBasePath = "D:/desktop/Res/tmp"
    js_lst = jsPathReader(dataBasePath)
    assert (len(js_lst) > 0)

    js_info_init = jsReader(js_lst[0])
    roi_init = points2Roi(js_info_init['shapes'][0]['points'])

    image_path = js_lst[0].replace('json', 'jpg')
    image_init = cv2.imread(image_path)
    # print(roi_init)
    # sys.exit()
    tracker = DSSTtracker(hog=True, fixed_windows=False, multi_scale=True)
    tracker.init(roi=roi_init, image=image_init)

    # init_bbox=Roi2Bbox(roi_init)
    # marked_image = MarkedBboxOnImage(image_init, init_bbox,color=(255,0,0),label='init')
    # ShowImage(marked_image)

    "test"
    testNum=1

    for testNum in range(10):
        js_info_test = jsReader(js_lst[testNum])
        image_path = js_lst[testNum].replace('json', 'jpg')
        image_test = cv2.imread(image_path, 1)
        roi_test = points2Roi(js_info_test['shapes'][0]['points'])
        label_bbox=Roi2Bbox(roi_test)

        image_test = MarkedBboxOnImage(image_test, label_bbox, label='label')


        detected_roi = tracker.update(image_test)
        detected_bbox = Roi2Bbox(detected_roi)
        marked_image = MarkedBboxOnImage(image_test, detected_bbox, label='Track')

        print("OP is: ",OPCal(label_bbox,detected_bbox))
        ShowImage(marked_image)



if __name__ == '__main__':
    test01()
