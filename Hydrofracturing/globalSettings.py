# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/1/11 1:52:40
# @File     : globalSettings.py

# @ref      : https://blog.csdn.net/weixin_34322964/article/details/78319449


# =========================配置全局共有变量（共两步）===========================
# ProjectName = "Hydrofracturing Analysis System"
ProjectName = "水力压裂分析"   # 第一步


def readSettingFile(request):
    return {
        "PROJECTNAME": ProjectName,  # 第二步
        # 增加全局变量时，同时在此返回变量
    }
