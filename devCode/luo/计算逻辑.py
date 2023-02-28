import math
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


#读取一个数据文件并处理基础数据
def calNum(filePath='.\测试数据1.csv'):
    data = pd.read_csv(filePath)
    #data = np.array(data)
    P = data['施工压力'][1:]
    Q = data['施工排量'][1:]
    sand = data['砂浓度'][1:]
    P = np.array(P)
    Q = np.array(Q)
    sand = np.array(sand)
    #需要输入的自变量
    d = 0.01            #射孔直径单位（m）
    D = 114.3/1000      #管径（m）
    L1 = 3000.0           #测深（m）
    L2 = 1000.0           #垂深（垂深）
    Cd = 0.6            #流量系数
    density = 1000.0      #流体密度（kg/m3）
    ISIP = 50           #瞬时停泵压力（MPa）
    a1 = 0              #分析起始位置
    a2 = 3000           #分析终点位置
    x = 3               #每簇开孔数

    #计算过程
    g = 9.8             #重力加速度
    N = []              #开启数目
    I = []
    press = []
    flow = []
    for j in range(a2-a1+1):
        P_wellhead = float(P[a1+j])
        Q_wellhead = float(Q[a1+j])
        sand_temp = float(sand[a1+j])
        roh = density + sand_temp
        P_hydraulic = roh * g * L2 / 10**6
        if Q_wellhead != 0:
            P_fline = (10.**(math.log10(Q_wellhead)*2+math.log10(0.1828)))*L1/1000*0.1       #0.1为减阻率90%（这个减阻率最好也能自己暗改，我这里默认给0.1）
        else:
            P_fline = 0
        P_fextend = ISIP
        for i in range(1,100):
            P_ffrac = 1.89*(Q_wellhead/i)**0.5
            P_fperf = P_wellhead + P_hydraulic - P_fline - P_fextend - P_ffrac
            n = (roh*2.24*10**-10*Q_wellhead*Q_wellhead/(P_fperf*d**4*Cd**2))**0.5
            if float(i)*float(x) > math.ceil(n):
                break
        N.append(math.ceil(n))
        I.append(i)
        flow.append(Q_wellhead)
        press.append(P_fperf)

    #结果展示
    print('预测开启孔个数为n=%d'%(N[-1]))

    #画图
    for i in range(1,31):   #射孔数目
        press_fperf = []
        for j in range(201):    #流量
            press_fperf.append((roh*2.24*10**-10*j/10*j/10/(i**2*d**4*Cd**2)))
        plt.plot([j/10 for j in range(201)],press_fperf,c='black',linewidth=0.5)
    plt.plot(flow,press)
    plt.ylim(0,50)
    plt.xlim(0,20)
    plt.xlabel('flow(m3/min)', fontsize = 15)
    plt.ylabel('P_fperf(MPa)', fontsize = 15)
    # plt.show()
    plt.savefig(r"static\SandPlugRiskEvaluation\retImgs\ret.svg", format="svg")
    plt.clf()


