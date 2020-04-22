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
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt


def createDataSet():
    dataSet = np.array([[1, 1, 'yes'],
                        [1, 1, 'yes'],
                        [1, 0, 'no'],
                        [0, 1, 'no'],
                        [0, 1, 'no']])
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet
