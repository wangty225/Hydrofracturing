# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/1/12 18:49:49
# @File     : urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home page'),
]