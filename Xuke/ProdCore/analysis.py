import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


def GRA_ONE(gray, m):
    # 用于计算特征关联分数值，以numpy array的形式返回
    # 列标准化
    gray = gray.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    std = gray.iloc[:, m]  # 参考序列
    # gray自身作为比较序列  # 比较序列

    # 与参考序列比较，相减
    gray = gray.apply(lambda x: np.abs(x - std))
    # 取出矩阵中最大值与最小值
    gray_max, gray_min = gray.values.max(), gray.values.min()

    gray = gray.apply(lambda x: ((gray_min + 0.5 * gray_max) / (x + 0.5 * gray_max)))
    # 求均值，得到灰色关联值
    res = np.average(gray, axis=0)  # 按行求均值
    return res[1:]


def Lasso_ONE(data):
    des = data.describe()
    r = des.T
    r = r[["min", "max", "mean", "std"]]
    np.round(r, 2)  # 保留2位小数
    np.round(
        data.corr(method="pearson"), 2
    )  # method={'pearson','spearman','kendall'},计算相关系数，相关分析

    model = Lasso(
        alpha=0.1,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
    )  # LASSO回归的特点是在拟合广义线性模型的同时进行变量筛选和复杂度调整，剔除存在共线性的变量
    model.fit(data.iloc[:, 1: data.shape[1]], data.iloc[:, 0])
    model_coef = pd.DataFrame(pd.DataFrame(model.coef_).T)
    RT = pd.DataFrame(model_coef.values.T)
    return RT.values[:, 0]


def Double_plot(dics, title, htmlFile, template="./data/Xuke/test"):
    # 绘图，并以html形式保存
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[1.0])

    for i, d in enumerate(dics):
        fig.add_trace(go.Bar(name=d["subName"], x=d["features"], y=d["values"]), row=1, col=1 + i)
    fig.update_layout(title_text=title)
    url = f"{template}/{htmlFile}.html"
    pio.write_html(fig, file=url)
