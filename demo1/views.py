from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "./demo1/page1.html")


def getPage2(request):
    if request.method=='GET':
        return render(request, './demo1/page2.html')
    return None


def index2(request):
    return render(request,
                  './temp.html',
                  {'source': '/demo1/getPage2'}
           )