import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import operator

# 特征字典
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}

def getDataSet():
    df = pd.read_csv('watermelon3_0_Ch.csv')
    dataSet = df.values[1:, 1:]  # 样本数据
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
    print(features)
    return dataSet, features

# 计算条件概率，计算先验概率
def cntProLap(dataSet, index, value, classLabel, N):      # 类别为classLabel，特征值为value，N为特征的N种取值
    extrData = dataSet[dataSet[:, -1] == classLabel]   #extrData表示标签为‘0’或者‘1’的
    cnt = 0
    for data in extrData:
        if data[index] == value:
            cnt += 1
    return (cnt + 1) / (float(len(extrData)) + N)


def naiveBayesClassifier(dataSet, features):
    dict = {}
    for feature in features:
        index = features.index(feature)      #index代表了特征的序号
        dict[feature] = {}
        if feature != '密度' and feature != '含糖率':  #为离散值
            featIList = featureDic[feature]  #该特征的所有特征值存在featIList中
            for value in featIList:
                PisCond = cntProLap(dataSet, index, value, '是', len(featIList))
                pNoCond = cntProLap(dataSet, index, value, '否', len(featIList))
                dict[feature][value] = {}
                dict[feature][value]["是"] = PisCond
                dict[feature][value]["否"] = pNoCond
        else:  #为连续值
            for label in ['是', '否']:
                dataExtra = dataSet[dataSet[:, -1] == label]
                extr = dataExtra[:, index].astype("float64")     #取出该特征的所有值（连续值，数字等），作为数组
                aver = extr.mean() #求平均
                var = extr.var()  #求方差

                labelStr = ""
                if label == '是':
                    labelStr = '是'
                else:
                    labelStr = '否'

                dict[feature][labelStr] = {}
                dict[feature][labelStr]["平均值"] = aver
                dict[feature][labelStr]["方差"] = var

    length = len(dataSet)    #样本数量
    classLabels = dataSet[:, -1].tolist()
    dict["好瓜"] = {}
    dict["好瓜"]['是'] = (classLabels.count('1') + 1) / (float(length) + 2)    #计算先验概率，这里的类别数为2，所以分母加2
    dict["好瓜"]['否'] = (classLabels.count('0') + 1) / (float(length) + 2)

    return dict

def NormDist(mean, var, xi):         #正态分布公式
    return exp(-((float(xi) - mean) ** 2) / (2 * var)) / (sqrt(2 * pi * var))


def predict(data, features, bayesDis):
    pGood = bayesDis['好瓜']['是']
    pBad = bayesDis['好瓜']['否']
    for feature in features:
        index = features.index(feature)
        if feature != '密度' and feature != '含糖率':
            pGood *= bayesDis[feature][data[index]]['是']
            pBad *= bayesDis[feature][data[index]]['否']
        else:
            # NormDist(mean, var, xi)
            pGood *= NormDist(bayesDis[feature]['是']['平均值'],
                              bayesDis[feature]['是']['方差'],
                              data[index])
            pBad *= NormDist(bayesDis[feature]['否']['平均值'],
                              bayesDis[feature]['否']['方差'],
                              data[index])

    retClass = ""
    if pGood > pBad:
        retClass = "好瓜"
    else:
        retClass = "坏瓜"

    return pGood, pBad, retClass

def calcAccRate(dataSet, features, bayesDis):
    cnt = 0.0
    for data in dataSet:
        _, _, pre = predict(data, features, bayesDis)
        if (pre == '好瓜' and data[-1] == '是') \
            or (pre == '坏瓜' and data[-1] == '否'):
            cnt += 1

    return cnt / float(len(dataSet))

dataSet, features = getDataSet()
dic = naiveBayesClassifier(dataSet, features)
print(f'字典树:\n{dic}')
p1, p0, pre = predict(dataSet[0], features, dic)
print(f"‘测1’为好瓜的概率：{p1}")
print(f"'测1'为坏瓜的概率： {p0}")
print(f"‘测1’为：{pre}")
print("准确率：", calcAccRate(dataSet, features, dic))
