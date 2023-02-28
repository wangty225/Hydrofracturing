import os
import time

import numpy as np
from django.shortcuts import render

from Hydrofracturing import settings

save_path_dir = r'data/SandPlugRiskEvaluation/excels/'


# Create your views here.
# /SandPlugRiskEvaluation/
def frame(request):
    fList = getFileList()
    if request.method == 'GET':
        return render(request, './SandPlugRiskEvaluation/frame.html', {'file_list': fList})

    file_object = request.FILES.get("uploadedFile")
    if file_object is None:
        return render(request, './SandPlugRiskEvaluation/frame.html', {'file_list': fList, 'message': '空文件！'})

    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # if not os.path.isfile(save_path_dir+file_object.name):
    f = open(save_path_dir + file_object.name, mode='wb')
    for chunk in file_object.chunks():
        f.write(chunk)
    f.close()

    return render(request, './SandPlugRiskEvaluation/frame.html', {'file_list': getFileList(), 'message': '文件上传成功！'})


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


# /SandPlugRiskEvaluation/chart
def getChart(request):
    path = request.GET.get('load')
    if path is not None:
        return json_response(getData(p=os.path.join(save_path_dir, request.GET.get('load'))))
    return json_response(getData())


def getFileList():
    L = []
    for root, dirs, files in os.walk(save_path_dir):
        for file in files:
            item = [time.ctime(os.path.getctime(os.path.join(save_path_dir, file))), file]
            # if os.path.splitext(file)[1] == '.csv':
            L.append(item)
    return L


# def read_img(request):
#     """
#     : 读取图片
#     :param request:
#     :return:
#     """
#     try:
#         data = request.GET
#         file_name = data.get("file_name")
#         imagepath = os.path.join(settings.STATIC_ROOT, "/Sandplug/images/{}".format(file_name))  # 图片路径
#         with open(imagepath, 'rb') as f:
#             image_data = f.read()
#         return HttpResponse(image_data, content_type="image/png")
#     except Exception as e:
#         print(e)
#         return HttpResponse(str(e))


import json
# from random import randrange

from django.http import HttpResponse


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

# def json_error(error_string="error", code=500, **kwargs):
#     data = {
#         "code": code,
#         "msg": error_string,
#         "data": {}
#     }
#     data.update(kwargs)
#     return response_as_json(data)


# JsonResponse = json_response
# JsonError = json_error

#
# def line_base() -> Line:
#     c = (
#         Line()
#         .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
#         .add_yaxis("商家A", [randrange(0, 100) for _ in range(6)])
#         .add_yaxis("商家B", [randrange(0, 100) for _ in range(6)])
#         .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="我是副标题"))
#         .dump_options_with_quotes()
#     )
#     return c


# class ChartView(APIView):
#     def get(self, request, *args, **kwargs):
#         return JsonResponse(json.loads(line_base()))
#
#
# class IndexView(APIView):
#     def get(self, request, *args, **kwargs):
#         return HttpResponse(content=open("./templates/index.html").read())
#
