# -*- coding=utf-8 -*-
import json

import matplotlib.markers
import numpy
import numpy as np
import time
import math
import os

import numpy.ma.core
import sympy
from django.http import HttpResponse
from numpy import ma
from sklearn import cluster, datasets
import pandas as pd
from sympy import *
import warnings

from collections import Counter
import matplotlib.pyplot as plt
from itertools import cycle

import matplotlib as mpl
import threading

# 汉字字体,优先使用仿宋，如果找不到则使用黑体
mpl.rcParams['font.sans-serif'] = ['SongTi', 'FangSong', 'KaiTi', 'SimHei']
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from SandPlugRiskEvaluation.core.utils.myLeastsq import *

RMSE_total_in = 0
RMSE_total_out = 0
RMSE_cnt = 0
RMSE_avg = 0.1
Rmse_Array = []

RMSE_Threshold = 3*RMSE_avg
KII_Threshold = 0.4
KI_Threshold = 0.8

index_P = 0
index_D = 1
index_T = 2
index_flag = 3
indexAP = [index_P, index_T]

save_path_dir = r'data/SandPlugRiskEvaluation/excels/'


def init(Slice=-1,
         SliceFrom=1560,
         p=r'..\..\data\SandPlugRiskEvaluation\excels\开发 JHW00525第6-1级 (二队 20200621）  施工数据.csv'):
    with open(p, encoding='gbk') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=2, usecols={1, 4})
        # print(data)
        #  data[:, 1]  取所有行，第二列的数据（砂浓度）
        start, end = getIndex(data[:, 1])
        # end2 = (end // 240) * 240
        # X = data[start:end2, :]
        X = data[start:end, :]

        # 以下一行代码，
        # 若注释掉即取全部约5000个数据，
        # 不注释则表示取油压数据 倒数第320 ~ 倒数第200 之间的120个数据
        if Slice != -1:
            X = X[SliceFrom:SliceFrom + Slice, :]  # 有1 的砂堵片段 SliceFrom=1560
            # X = X[:98, :]
            # X = X[3050: 3290]  # 有2 可能砂堵
            # X = X[1860:3800]  # 有4 砂堵片段
            pass
        # print(X)

        # X = np.insert(X, 2, values=np.arange(1, len(X[:, 1])+1), axis=1)
        X = np.c_[X, np.arange(1, len(X[:, 1]) + 1).reshape(-1, 1)]  # 第[2]列T

        # column_names = ['P', 'D', 'T', 'LogP', 'LogT', 'e']
        # df = pd.DataFrame(X, columns=column_names, index=None)
        # df.to_csv(save_path)

        X = np.c_[X, np.zeros(len(X[:, 1])).reshape(-1, 1)]  # 第[6]列风险等级

        return X


'''           油压P   砂浓度D  时间T      LogP      LogT       e
    X:        62.39   300.0   1.0    5.963243  0.000000  0.000438
              62.46   300.0   2.0    5.964861  1.000000 -0.000462
              62.42   300.0   3.0    5.963936  1.584963 -0.000627
              62.44   300.0   4.0    5.964399  2.000000  0.000395
              62.40   300.0   5.0    5.963474  2.321928  0.001905
              ......
'''


# 绘图
def myPlt(Y, X, ylabel, xlabel, title):
    plt.scatter(X, Y, marker='.')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


# 返回砂浓度不为0时，对应油压的最左和最右索引值
# 油压 x[start]=0  x[end]!=0
def getIndex(x):
    s = 0
    for i in range(0, len(x)):
        if x[i] == 0:
            s += 1
        else:
            break

    e = len(x) - 1
    for i in range(len(x) - 1, 0, -1):
        if x[i] == 0:
            e -= 1
        else:
            break
    # print(s, e)
    return s, e


def getRangeFromClassMembers(classMembers):
    # lis = classMembers.tolist()
    lis = classMembers
    if lis.count(True) >= 0:
        return lis.index(True), lis.count(True)
    return 0, 0


def callAlert(X, K, Partition, classMembers, RMSE=0):
    level = 0
    if K >= KI_Threshold:
        level = 1
    elif K >= KII_Threshold:
        level = 2
    elif RMSE >= RMSE_Threshold and K > 0:
        level = 3
    else:
        level = 4

    message = ['',
               'I级砂堵预警：发生砂堵！ 时间范围：',
               'II级砂堵预警：出现砂堵征兆！ 时间范围：',
               'III级砂堵预警：超过历史均方差值*3！ 时间范围：',
               'IV级砂堵预警：无异常！ 时间范围：']
    if level != 4:
        warnings.warn(message[level] + str(Partition[0]) + '~' + str(Partition[1]))
    # X[classMembers, index_flag] = level
    for item in range(0, len(classMembers)):
        if classMembers[item] == True:
            X[item, index_flag] = level
    return level


def y_update_scale_value(val, position):
    if val == 1:
        return 'IV'
    elif val == 2:
        return 'III'
    elif val == 3:
        return 'II'
    elif val == 4:
        return 'I'
    else:
        return ''


def getMyMask(mk) -> np.ma.core.bool_:
    if isinstance(mk, np.ma.core.bool_):
        return mk
    for i in range(1, len(mk)):
        if mk[i] == False and mk[i - 1] == True:
            mk[i - 1] = False
    return mk


def getRMSE_Threshold(dataArray):
    df = pd.DataFrame(dataArray, columns=['value'])
    # 计算均值
    u = df['value'].mean()
    #### # print("均值u:", u)
    ##### # 计算标准差
    std = df['value'].std()
    ##### # print("标准差std:", std)
    return u + 3 * std


def myFit(X, lsqX, lsqY, classMembers):
    global RMSE_total_in, RMSE_total_out
    global RMSE_cnt, RMSE_avg, RMSE_Threshold
    global Rmse_Array
    p1 = [0, 1000]  # 最小二乘法一次线性
    para1 = leastsq(error1, p1, args=(lsqX, lsqY))
    ######### # print("y = %.5f*x + %.5f" % (para1[0][0], para1[0][1]))
    y_fitted = Fun1(para1[0], lsqX)
    # plt.plot(lsqX, y_fitted, c='r', label='1次拟合')
    partStart, partLength = getRangeFromClassMembers(classMembers)
    ###### # print("partIndex: [", partStart, " , ", partStart + partLength, ")")  # 返回为索引值，各加一才为对应横坐标的值
    cur_RMSE = getRMSE(lsqY, y_fitted)
    Rmse_Array += [cur_RMSE]
    callAlert(X, para1[0][0], [partStart + 1, partStart + partLength + 1], classMembers, cur_RMSE)
    if cur_RMSE <= RMSE_Threshold:  # 一次拟合效果较好，IV级风险
        # Rmse_Array += [cur_RMSE]
        RMSE_total_in += cur_RMSE
        RMSE_cnt += 1
        # RMSE_avg = (RMSE_total_in * 0.99865 + RMSE_total_out * 0.00135) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        RMSE_avg = (RMSE_total_in + RMSE_total_out) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        # RMSE_avg = (RMSE_total_in) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        RMSE_Threshold = 3 * RMSE_avg
    ###### # print("3*miu:", RMSE_Threshold)
    # if len(Rmse_Array)>50:
    # RMSE_Threshold = getRMSE_Threshold(Rmse_Array)
    # print("my:", RMSE_Threshold)
    if cur_RMSE > RMSE_Threshold:
        RMSE_total_out += cur_RMSE
        RMSE_cnt += 1
        # RMSE_avg = (RMSE_total_in * 0.99865 + RMSE_total_out * 0.00135) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        RMSE_avg = (RMSE_total_in + RMSE_total_out) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        # RMSE_avg = (RMSE_total_in) / RMSE_cnt * 0.5 + RMSE_avg * 0.5
        RMSE_Threshold = 3 * RMSE_avg
        # if len(Rmse_Array)>50:
        # RMSE_Threshold = getRMSE_Threshold(Rmse_Array)
        p2 = [0, 0.01, 1000]  # 拟合的初始参数设置
        para2 = leastsq(error2, p2, args=(lsqX, lsqY))  # 进行拟合
        ####### # print("y = %.5f*x^2 + %.5f*x + %.5f" % (para2[0][0], para2[0][1], para2[0][2]))
        y_fitted = Fun2(para2[0], lsqX)  # 画出拟合后的曲线
        # plt.plot(lsqX, y_fitted, c='g', label='2次拟合')

        # 分左右段线性拟合
        x = Symbol('x')
        ret = solve([para2[0][0] * 2 * x + para2[0][1]], x)  # 导函数等于0的点
        ######## # print("分段区间t", lsqX.min(), lsqX.max())
        ######## # print("斜率为0的点", ret)
        ret[x] = np.float32(ret[x])
        retx = numpy.array([sympy.N(i) for i in ret])
        ######## # print(retx)
        rety = para2[0][0] * ret[x] ** 2 + para2[0][1] * ret[x] + para2[0][2]
        lsq_x_l = list(lsqX[lsqX <= ret[x]])
        lsq_x_r = list(lsqX[lsqX >= ret[x]])

        lsq_y_l = list(lsqY[lsqX <= ret[x]])
        lsq_y_r = list(lsqY[lsqX >= ret[x]])

        if math.ceil(lsqX.min()) < ret[x] < math.floor(lsqX.max()):
            lsq_x_l.append(ret[x])
            lsq_x_r = [ret[x], ] + lsq_x_r
            lsq_y_l.append(rety)
            lsq_y_r = [rety, ] + lsq_y_r
        else:
            return

        #######    # print("左半分段x", lsq_x_l, "右半分段x", lsq_x_r)
        #######    # print("左半分段y", lsq_y_l, "右半分段y", lsq_y_r)
        lsq_x_l = np.array(lsq_x_l)
        lsq_x_r = np.array(lsq_x_r)
        lsq_y_l = np.array(lsq_y_l)
        lsq_y_r = np.array(lsq_y_r)

        if len(lsq_x_l) > 1:
            class_members2 = []

            for i in range(0, math.floor(lsq_x_l[0])):
                class_members2.append(False)
            for i in range(math.floor(lsq_x_l[0]), math.floor(lsq_x_l[len(lsq_x_l) - 1]) + 1):
                class_members2.append(True)
            for i in range(math.floor(lsq_x_l[len(lsq_x_l) - 1]) + 1, len(classMembers)):
                class_members2.append(False)
            myFit(X, lsq_x_l, lsq_y_l, class_members2)
            # callAlert(X, para1[0][0], [lsq_x_l[0], lsq_x_l[len(lsq_x_l) - 1]], class_members2)

        if len(lsq_x_r) > 1:
            class_members3 = []
            for i in range(0, math.floor(lsq_x_r[0])):
                class_members3.append(False)
            for i in range(math.floor(lsq_x_r[0]), math.floor(lsq_x_r[len(lsq_x_r) - 1]) + 1):
                class_members3.append(True)
            for i in range(math.floor(lsq_x_r[len(lsq_x_r) - 1]) + 1, len(classMembers)):
                class_members3.append(False)
            myFit(X, lsq_x_r, lsq_y_r, class_members3)
            # callAlert(X, para1[0][0], [lsq_x_r[0], lsq_x_r[len(lsq_x_r) - 1]], class_members3)


def run(X_raw, tt=1860, Slice=360):
    if tt < Slice:
        return
    if tt > len(X_raw):
        return -1
    # X = init(Slice=-1, updateFlag=True)
    X = X_raw[tt + 1 - Slice:tt + 1, :]
    # Case 1: AP聚类算法, 该算法对应的两个参数
    t0 = time.time()
    indexClass = ['P', 'T']
    affinity_propagation = cluster.AffinityPropagation(
        damping=.8, preference=-200, random_state=0).fit(X[:, indexAP])
    t = time.time() - t0

    # 绘AP聚类图
    cluster_centers_indices = affinity_propagation.cluster_centers_indices_  # 预测出的中心点的索引，如[123,23,34]
    labels = affinity_propagation.labels_  # 预测出的每个数据的类别标签,labels是一个NumPy数组
    n_clusters_ = len(cluster_centers_indices)
    for k in range(n_clusters_):
        class_members = labels == k

        # 采用（左闭右开）的方式处理数据分段
        for i in range(1, len(class_members) - 1):
            if class_members[i] == True and class_members[i - 1] == False:
                class_members[i - 1] = True

        class_members2 = []
        for i in range(0, tt + 1 - Slice):
            class_members2.append(False)
        for i in range(len(class_members)):
            class_members2.append(class_members[i])
        for i in range(tt + 1, len(X_raw)):
            class_members2.append(False)
        lsq_x = X[class_members, indexAP[1]]
        lsq_y = X[class_members, indexAP[0]]
        myFit(X_raw, lsq_x, lsq_y, class_members2)

    X = X_raw[tt + 1 - Slice:tt + 1, :]
    # thread = threading.Thread(target=getSvg, args=(X, )).start()
    # return X[-1, index_flag],

    # def getSvg(X):
    #     plt.close('all')  # 关闭所有的图形
    #     plt.figure(1)  # 产生一个新的图形
    #     plt.clf()  # 清空当前的图形
    from matplotlib.ticker import FuncFormatter
    plt.switch_backend('agg')
    # colori = 'kgbyr'
    axes = plt.gca()
    axes2 = axes.twinx()
    axes2.yaxis.set_major_formatter(FuncFormatter(y_update_scale_value))
    data_flag = 5 - X[:, index_flag]
    data = X[:, index_P]
    fMask1 = np.ma.masked_outside(data_flag, 0.5, 1.5)
    fMask2 = np.ma.masked_outside(data_flag, 1.5, 2.5)
    fMask3 = np.ma.masked_outside(data_flag, 2.5, 3.5)
    fMask4 = np.ma.masked_outside(data_flag, 3.5, 4.5)
    m1 = ma.MaskedArray(data_flag, mask=(numpy.ma.getmask(fMask1)))
    m2 = ma.MaskedArray(data_flag, mask=(numpy.ma.getmask(fMask2)))
    m3 = ma.MaskedArray(data_flag, mask=(numpy.ma.getmask(fMask3)))
    m4 = ma.MaskedArray(data_flag, mask=(numpy.ma.getmask(fMask4)))
    d1 = np.ma.MaskedArray(data, getMyMask(numpy.ma.getmask(fMask1)))
    d2 = np.ma.MaskedArray(data, getMyMask(numpy.ma.getmask(fMask2)))
    d3 = np.ma.MaskedArray(data, getMyMask(numpy.ma.getmask(fMask3)))
    d4 = np.ma.MaskedArray(data, getMyMask(numpy.ma.getmask(fMask4)))

    axes.plot(X[:, indexAP[1]], data, '-', c='k', linewidth=.5)
    axes.plot(X[:, indexAP[1]], d1, '-', c='b', linewidth=.8)
    axes.plot(X[:, indexAP[1]], d2, '-', c='g', linewidth=.8)
    axes.plot(X[:, indexAP[1]], d3, '-', c='y', linewidth=.8)
    axes.plot(X[:, indexAP[1]], d4, '-', c='r', linewidth=.8)

    axes.set_ylabel('P')
    axes.grid(linestyle='-.')
    axes.set_ylim([40, 70])
    axes2.set_ylim([0, 9])
    axes2.plot(X[:, indexAP[1]], data_flag, '-', c='k', linewidth=.5, label='')
    axes2.plot(X[:, indexAP[1]], m1, '-', c='b', linewidth=1, label='无砂堵风险IV')
    axes2.plot(X[:, indexAP[1]], m2, '-', c='g', linewidth=1, label='低砂堵风险III')
    axes2.plot(X[:, indexAP[1]], m3, '-', c='y', linewidth=1, label='中砂堵风险II')
    # axes2.plot(X[:, indexAP[1]], m3, '-^', c='y', linewidth=1, label='中高砂堵风险II')
    axes2.plot(X[:, indexAP[1]], m4, '-', c='r', linewidth=1, label='高砂堵风险I')
    # axes2.plot(X[:, indexAP[1]], m4, '-^', c='r', linewidth=1, label='高砂堵风险I')
    axes2.set_ylabel('风险等级')
    plt.title("预警结果可视化")
    plt.legend(loc='upper left', fontsize=10)
    # plt.show()
    plt.savefig(r"..\..\static\SandPlugRiskEvaluation\retImgs\ret.svg", format="svg")
    # plt.savefig(r"static\SandPlugRiskEvaluation\retImgs\ret.svg", format="svg")
    plt.clf()

    # column_names = ['P', 'D', 'T', 'LogP', 'LogT', 'e', 'flag']
    # df = pd.DataFrame(X, columns=column_names, index=None)
    # df.to_csv(r'.\data.csv')
    # print(getRMSE_Threshold(Rmse_Array))
    return X[-1, index_flag]


# Create your views here.
def response_as_json(data):
    json_str = json.dumps(data)
    response = HttpResponse(
        json_str,
        content_type="application/json",
    )
    response["Access-Control-Allow-Origin"] = "*"
    return response


def json_response(data, code=200):
    data = {
        "code": code,
        "msg": "success",
        "data": data,
    }
    return response_as_json(data)

# // 入口
def test(request):
    X_raw = init(Slice=-1, p=save_path_dir + request.POST.get("fileName"))
    startSec = int(request.POST.get("startSec"))
    ret = run(X_raw, tt=startSec, Slice=360)
    return json_response(ret)


if __name__ == '__main__':
    X_raw = init(Slice=-1)
    startSec = 1658-154
    startSec = 1658
    Slice = 100
    # for i in range(startSec, startSec+1):
    # # for i in range(startSec, len(X_raw)):
    #     X0 = X_raw[startSec+1-Slice:startSec+1, :].copy()
    #     threading.Thread(target=run, args=(X0, startSec, Slice)).start()
    level = run(X_raw, tt=startSec, Slice=300)
    print(level)
