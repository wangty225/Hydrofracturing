from django.http import HttpRequest
from django.shortcuts import redirect, render
from Xuke.ProdCore.analysis import GRA_ONE, Lasso_ONE, Double_plot
from pandas import read_csv

# Create your views here.


def prod_login(request):
    """用户登录"""
    if request.method == "GET":
        return render(request, "./Xuke/login.html")
        # return render(request, "fig.html")

    user = request.POST.get("user")
    pwd = request.POST.get("pwd")
    if user == "root" and pwd == "123":
        return redirect("/Xuke/prod/home/")
    # 若用户名或密码输入错误
    return render(request, "./Xuke/login.html", {"error_msg": "若用户名或密码输入错误!"})


def prod_home(request):
    """主页"""
    return render(request, "./Xuke/home.html")


def prod_open(request):
    """打开文件夹"""
    file = "./data/Xuke/test/geology.csv" # 这里使用了相对路径
    from csv import DictReader

    with open(file, mode="r", encoding="utf") as f:
        data = list(DictReader(f))
        
    return render(request, "./Xuke/open.html", {"data": data})


def prod_grey(request):
    ''' 灰色关联分析 '''
    path = "./data/Xuke/test" # 这里使用了相对路径
    geo = "geology.csv"
    eng = "Engineering.csv"
    
    ''' 2.1 同时对地质、工程两个csv灰色关联分析，绘制柱状图，保存结果为html '''
    dics = []
    for file in [geo, eng]:
        data = read_csv(f"{path}/{file}") # geology.csv
        features = data.columns[1:] # 特征名称
        values = GRA_ONE(data, m=0) # 特征关联分数值
        dic = {
            "features": features,
            "values": values
        }
        dics.append(dic)

    dics[0]["subName"] = "地质因素评估指标"
    dics[1]["subName"] = "工程因素评估指标"
    Double_plot(dics, "灰色关联分析", "grey")

    return render(request, "./Xuke/grey.html")


def prod_Lasso(request):
    ''' Lasso 分析 '''
    path = "./data/Xuke/test" # 这里使用了相对路径
    geo = "geology.csv"
    eng = "Engineering.csv"
    
    ''' 2.1 同时对地质、工程两个csvLasso 分析，绘制柱状图，保存结果为html '''
    dics = []
    for file in [geo, eng]:
        data = read_csv(f"{path}/{file}") # geology.csv
        features = data.columns[1:] # 特征名称
        values = Lasso_ONE(data) # 特征关联分数值
        dic = {
            "features": features,
            "values": values
        }
        dics.append(dic)

    dics[0]["subName"] = "地质因素评估指标"
    dics[1]["subName"] = "工程因素评估指标"
    Double_plot(dics, "Lasso 分析", "Lasso")

    return render(request, "./Xuke/Lasso.html")


# 对应方法二：temp.html是模板，source是主体iframe里的内容
def prod(request):
    return render(request, "./temp.html", {"source" : '/Xuke/prod/login'} )
