# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/1/10 07:06:44
# @File     : views.py

from django.http import HttpResponse
from django.shortcuts import render

from Hydrofracturing import settings


def index(request):
    return render(request, './index/index.html')


# def i(request):
#     return render(request, './index/123.html')

