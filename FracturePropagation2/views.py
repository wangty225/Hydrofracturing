import os
import time

from django.http import HttpResponse
from django.shortcuts import render

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
    if request.method == 'GET':
        return render(request, './FracturePropagation/base.html', {'file_list': fList})
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

    return render(request, './FracturePropagation/base.html', {'file_list': getFileList(save_path_dir), 'message': '文件上传成功！'})


