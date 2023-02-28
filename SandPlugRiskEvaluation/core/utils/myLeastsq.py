import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def Fun1(p, x):  # 定义拟合函数形式
    a0, a1 = p
    return a0 * x + a1


def Fun2(p, x):  # 定义拟合函数形式
    a0, a1, a2 = p
    return a0 * x ** 2 + a1 * x + a2


def Fun3(p, x):  # 定义拟合函数形式
    a0, a1, a2, a3 = p
    return a0 * x ** 3 + a1 * x ** 2 + a2 * x + a3


def error1(p, x, y):  # 拟合残差
    return Fun1(p, x) - y


def error2(p, x, y):  # 拟合残差
    return Fun2(p, x) - y


def error3(p, x, y):  # 拟合残差
    return Fun3(p, x) - y


def getMSE(y, y_fitted):
    sum_deltaY2 = 0
    for i in range(0, len(y)):
        sum_deltaY2 += (y_fitted[i] - y[i]) ** 2
    MSE = sum_deltaY2/len(y)
    print("MSE: ", MSE)
    return MSE


def getRMSE(y, y_fitted):
    sum_deltaY2 = 0
    for i in range(0, len(y)):
        sum_deltaY2 += (y_fitted[i] - y[i]) ** 2
    RMSE = math.sqrt(sum_deltaY2/len(y))
   ##### # print("RMSE: ", RMSE)
    return RMSE


def dFun3(p, x):
    a0, a1, a2, a3 = p
    return 3*a0 * x ** 2 + 2*a1 * x + a2


def dFun2(p, x):
    a0, a1, a2 = p
    return 2*a0 * x + a1


def main():
    x = np.linspace(-10, 10, 100)  # 创建时间序列
    noise = np.random.randn(len(x))  # 创建随机噪声

    p_value = [-2, 5, 10]  # 原始数据的参数
    y = Fun2(p_value, x) + noise * 2  # 加上噪声的序列
    p0 = [0, -0.01, 100]  # 拟合的初始参数设置
    para = leastsq(error2, p0, args=(x, y))  # 进行拟合
    print(para)
    y_fitted = Fun2(para[0], x)  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'o', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()

    p_value = [1, -2, 5, 10]  # 原始数据的参数
    y = Fun3(p_value, x) + noise * 2  # 加上噪声的序列
    p0 = [0.5, -0.01, 5, 10]  # 拟合的初始参数设置
    para = leastsq(error3, p0, args=(x, y))  # 进行拟合
    print(para)
    y_fitted = Fun3(para[0], x)  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'o', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()

    p_value = [0.05, 10]  # 原始数据的参数
    y = Fun1(p_value, x) + noise * 0.2  # 加上噪声的序列
    p0 = [0.5, -0.01]  # 拟合的初始参数设置
    para = leastsq(error1, p0, args=(x, y))  # 进行拟合
    print(para)
    y_fitted = Fun1(para[0], x)  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'o', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()