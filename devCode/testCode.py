# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/8/25 18:42:39
# @File     : testCode.py

l = [0 for _ in range(10)]

# l[0:2]=1  # 错，不可以修改副本
print(l[0:2])

for i in range(0, 2):
    l[i]=1
print(l)