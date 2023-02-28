import json
import os
import time

import numpy as np
from django.http import HttpResponse
from django.shortcuts import render

from CalPerforationNum.core.calNum import calNum

save_path_dir = r'data/CalPerforationNum/excels/'


def getFileList(save_path):
    L = []
    for root, dirs, files in os.walk(save_path):
        for file in files:
            item = [time.ctime(os.path.getctime(os.path.join(save_path, file))), file]
            # if os.path.splitext(file)[1] == '.csv':
            L.append(item)
    return L


# Create your views here.
def index(request):
    fList = getFileList(save_path_dir)
    if request.method == 'GET':
        return render(request, './CalPerforationNum/index.html', {'file_list': fList, 'num': 0})

    file_object = request.FILES.get("uploadedFile")
    if file_object is None:
        return render(request, './CalPerforationNum/index.html', {'file_list': fList, 'message': '空文件！', 'num': 0})

    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # if not os.path.isfile(save_path_dir+file_object.name):
    f = open(save_path_dir + file_object.name, mode='wb')
    for chunk in file_object.chunks():
        f.write(chunk)
    f.close()

    return render(request, './SandPlugRiskEvaluation/frame.html',
                  {'file_list': getFileList(save_path_dir), 'message': '文件上传成功！', 'num': 0})


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


def cal(request):
    if request.method == 'POST':
        arg1 = request.POST.get("arg1")
        arg2 = request.POST.get("arg2")
        arg3 = request.POST.get("arg3")
        arg4 = request.POST.get("arg4")
        arg5 = request.POST.get("arg5")
        fileName = request.POST.get("fileName")
        print(arg1, arg2, arg3, arg4, arg5, fileName)
        # ret=调用函数
        fileName = save_path_dir + fileName
        nums = calNum(fileName)
        ret = json.dumps({"num": nums})
        # 图片保存在static/CalPerforationNum/retImgs/ret.svg
        # ret = 4
        return json_response(ret)


# //chart
def getChart(request):
    path = request.GET.get('load')
    if path is not None:
        return json_response(getData(p=os.path.join(save_path_dir, request.GET.get('load'))))
    return json_response(getData())


def getData(p=r'.\SandPlugRiskEvaluation\data\you\1\1.csv'):
    # p=r'.\SandPlugRiskEvaluation\data\有\1\开发 JHW00525第6-1级 (二队 20200621）  施工数据.csv'
    # p = r'.\SandPlugRiskEvaluation\data\you\1\1.csv'

    with open(p, encoding='gbk') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=2, usecols={1, 2, 4})
        start, end = getIndex(data[:, 2])
        # print(start, end)
        X = data[start:end, :]
        X = np.insert(X, 0, values=np.arange(1, len(X[:, 1]) + 1), axis=1)
        ret = []
        for i in range(0, len(X[0])):
            ret.append(X[:, i].tolist())
        return ret



def getIndex(x):
    s = 0
    for i in range(0, len(x)):
        if float(x[i]) == 0:
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
