# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
data = pd.read_csv('西瓜数据集3.0alpha.csv')

data = data.values
X = data[:, 0:2]
y = data[:, 2]

# 查看数据分布
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Good')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.legend()
plt.show()

# 构建决策树
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
tree_clf.fit(X, y)

# 决策边界绘制函数
def plot_decision_boundary(model, x):
    # 生成网格点坐标矩阵,得到两个矩阵
    M, N = 500, 500
    x0, x1 = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 0.6, N))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    z = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#90CAF9'])
    plt.contourf(x0, x1, z, cmap=custom_cmap, alpha=0.8)

# 绘制决策边界
plot_decision_boundary(tree_clf, X)
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Good')
plt.xlabel('Density')
plt.ylabel('Sugar Content')
plt.legend()
plt.show()

# 查看特征重要性
print("Feature Importances:", tree_clf.feature_importances_)