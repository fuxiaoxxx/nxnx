# -*- coding: utf-8 -*-
# 西瓜数据集3.0a分类性能图和决策边界
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取西瓜数据集3.0a
data = pd.read_csv('3.0a.txt', delimiter=' ')
X = data[['密度', '含糖率']]
y = data['好瓜']

# 使用Z-score标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义SVM模型参数
param_grid = {
    'C': [1, 10, 100],  # 正则化参数的适量取值
    'gamma': [0.1, 1],  # 核函数系数的适量取值
    'kernel': ['linear', 'rbf']  # 线性核和高斯核（RBF核）
}

# 遍历每种参数组合并绘制决策边界
for params in ParameterGrid(param_grid):
    # 创建SVM模型并设置参数
    model = svm.SVC(C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
    model.fit(X_scaled, y)

    # 计算并输出当前参数组合的准确率
    accuracy = accuracy_score(y, model.predict(X_scaled))
    print(f"当前参数组合: C={params['C']}, gamma={params['gamma']}, kernel={params['kernel']}, 准确率={accuracy:.2f}")

    # 生成网格点
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # 使用当前模型预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和散点图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50)
    plt.xlabel('密度 (标准化)')
    plt.ylabel('含糖率 (标准化)')
    plt.title(f'SVM 决策边界 (C={params["C"]}, gamma={params["gamma"]}, kernel={params["kernel"]})\n准确率: {accuracy:.2f}')
    plt.show()
