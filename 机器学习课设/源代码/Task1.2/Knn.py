# # -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def knn(train, test, num):
#     # train,test,num分别为训练集，测试集，近邻个数
#     output = []  # 预测分类结果
#     m, n = len(train), len(test)
#     for i in range(n):
#         dist_ij = []
#         for j in range(m):
#             d = np.linalg.norm(test[i, :] - train[j, :-1])  # 求范数
#             dist_ij.append((j, d))
#         id_min = sorted(dist_ij, key=lambda x: x[1])[:num]
#         rate = [train[i[0], -1] for i in id_min]
#         if sum(rate) / num >= 0.5:  # 当两类得票数相等时，优先划分为正例
#             output.append(1)
#         else:
#             output.append(0)
#
#     return output
#
# # 加载数据集
# data = pd.read_csv('西瓜数据集3.0alpha.csv',skiprows=0)
# print(data)
# data = data.values
# print(data)
# print(data.shape)
#
# # 生成网格点
# a = np.arange(0, 1.01, 0.01)
# b = np.arange(0, 0.61, 0.01)
# x, y = np.meshgrid(a, b)
# k = 5  # 近邻个数
#
# # 使用KNN进行分类
# z = knn(data, np.c_[x.ravel(), y.ravel()], k)
# z = np.array(z).reshape(x.shape)
#
# # 可视化
# fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.contourf(x, y, z, cmap=plt.cm.winter, alpha=.6)
# label_map = {1: 'good', 0: 'bad'}
# ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.PuBuGn, edgecolors='k')
# ax.set_xlim(0, 1.0)
# ax.set_ylim(0, 0.6)
# ax.set_ylabel('含糖率')
# ax.set_xlabel('密度')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# ax.set_title(' %s-近邻分类器' % k)
# plt.show()
#
#
#
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def knn(train, test, num):
    # train, test, num 分别为训练集，测试集，近邻个数
    output = []  # 预测分类结果
    m, n = len(train), len(test)
    for i in range(n):
        dist_ij = []
        for j in range(m):
            d = np.linalg.norm(test[i, :] - train[j, :-1])  # 求范数
            dist_ij.append((j, d))
        id_min = sorted(dist_ij, key=lambda x: x[1])[:num]
        rate = [train[i[0], -1] for i in id_min]
        if sum(rate) / num >= 0.5:  # 当两类得票数相等时，优先划分为正例
            output.append(1)
        else:
            output.append(0)
    return output

# 加载数据集
data = pd.read_csv('西瓜数据集3.0alpha.csv', skiprows=0)
print(data)
data = data.values
print(data)
print(data.shape)

# 生成网格点
a = np.arange(0, 1.01, 0.01)
b = np.arange(0, 0.61, 0.01)
x, y = np.meshgrid(a, b)
k = 5  # 近邻个数

# 使用KNN进行分类
z = knn(data, np.c_[x.ravel(), y.ravel()], k)
z = np.array(z).reshape(x.shape)

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# 绘制分类边界
ax.contourf(x, y, z, cmap=plt.cm.winter, alpha=.6)

# 绘制训练数据点并标记标签
label_map = {1: 'good', 0: 'bad'}
scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=plt.cm.PuBuGn, edgecolors='k')

# 添加图例
handles, labels = scatter.legend_elements()
ax.legend(handles, ['Good', 'Bad'], title="Label")

# 设置坐标轴标签
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.6)
ax.set_ylabel('含糖率')
ax.set_xlabel('密度')

# 设置标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax.set_title(f'{k} - 近邻分类器')

# 显示图形
plt.show()
