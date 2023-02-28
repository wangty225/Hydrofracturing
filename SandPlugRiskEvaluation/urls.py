# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/1/12 13:15:38
# @File     : urls.py

from django.urls import path
from . import views
from SandPlugRiskEvaluation.core import eval

urlpatterns = [
    path('', views.frame, name='SandPlugRiskEvaluation index'),
    path(r"chart", views.getChart, name='SandPlugRiskEvaluation'),
    # path(r'file_list', views.getFileList, name='getFileList'),
    path(r'eval', eval.test, name='eval'),
]
