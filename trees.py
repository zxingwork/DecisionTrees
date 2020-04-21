#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/4/21 6:50 下午
# software: PyCharm
import numpy as np


def clcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCount = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(currentLabel[key]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt

