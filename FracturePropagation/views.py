import json
import os
import sys
import time

from django.http import JsonResponse
from django.shortcuts import render,HttpResponse

from FracturePropagation.core.core import input_choose_y, input_choose_n

save_path_dir = r'data/FracturePropagation/excels/'
# save_path_dir = r'data/SandPlugRiskEvaluation/excels/'

def getFileList(save_path):
    L = []
    if not os.path.exists(save_path):
        return []
    for root, dirs, files in os.walk(save_path):
        for file in files:
            item = [time.ctime(os.path.getctime(os.path.join(save_path, file))), file]
            # if os.path.splitext(file)[1] == '.csv':
            L.append(item)
    return L


# Create your views here.
def base(request):
    fList = getFileList(save_path_dir)
    global flag
    if request.method == 'GET':
        flag=True
        return render(request, './FracturePropagation/base.html', {'file_list': fList,
                                                                   'logs':'当前未载入测试文件，请导入！',
                                                                   'flag':flag
                                                                   })
    file_object = request.FILES.get("uploadedFile")
    if file_object is None:
        return render(request, './FracturePropagation/base.html', {'file_list': fList, 'message': '空文件！'})

    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    # if not os.path.isfile(save_path_dir+file_object.name):
    f = open(save_path_dir + file_object.name, mode='wb')
    for chunk in file_object.chunks():
        f.write(chunk)
    f.close()
    output = sys.stdout
    # outputfile = open('./out.log', 'w')
    outputfile=open(r'static/FracturePropagation/out.txt','w')
    sys.stdout = outputfile
    global name
    name=file_object.name.strip('.xlsx')
    input_choose_y(filename=name)
    outputfile.close()
    flag=False
    sys.stdout = output
    file_txt=open(r'static/FracturePropagation/out.txt','r')
    logs=file_txt.read()
    return render(request, './FracturePropagation/base.html', {'file_list': getFileList(save_path_dir),
                                                               'message': '文件上传成功,请及时下载分析报告！',
                                                               'logs':logs,
                                                               'flag':flag,
                                                               'name':name
                                                               })

# /FracturePropagation/download
def download(request):
    if flag==False:
        file = open(r'static/FracturePropagation/result/' + name + '/分析报告_' + name + '.docx', 'rb')
        # response = HttpResponse(file)
        # response['Content-Type'] = 'application/msword'
        # response=HttpResponse(file.read(),content_type='application/msword')
        # response['Content-Disposition'] = 'attachment;filename="分析报告_' + name + '.docx"'
        response=HttpResponse()
        response['Content-Type']='application/msword'
        response['Content-Disposition'] = 'attachment;filename="分析报告_' + name + '.docx"'
        response.content=file.read()
        return response
    else:
        # file = open(r'static/FracturePropagation/result/' + name + '/分析报告_' + name + '.docx', 'rb')
        file=open(r'static/FracturePropagation/result/test/分析报告_test.docx','rb')
        response = HttpResponse()
        response['Content-Type'] = 'application/msword'
        response['Content-Disposition'] = 'attachment;filename="分析报告_test.docx'

        response.content = file.read()
        return response

def calculate(request):
    data=request.body.decode("utf-8") # request.body 显示前端发送的json数据，decode是设置编码格式
    # json_data=json.loads(data) # 把json数据转换成字典
    # print(json_data)
    output=sys.stdout
    outputfile=open(r'static/FracturePropagation/out.txt','w')
    sys.stdout=outputfile

    input_choose_n(data_json=data)

    outputfile.close()
    sys.stdout=output
    file_txt = open(r'static/FracturePropagation/out.txt', 'r')
    logs = file_txt.read()

    return JsonResponse({"msg":"传递参数成功！",
                         "logs":logs,
                         "message":"计算完成，请及时下载分析报告！",
                         "flag":False,
                         "name":"test"
                         })

