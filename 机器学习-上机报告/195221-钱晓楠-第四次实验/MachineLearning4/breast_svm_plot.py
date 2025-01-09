# -*- coding: utf-8 -*-
# 乳腺癌诊断分类
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据集
data = pd.read_csv("./data.csv")

# 数据清洗
data.drop("id", axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# 抽取30%的数据作为测试集
train, test = train_test_split(data, test_size=0.3, random_state=42)

# 分离特征和标签
train_X = train.drop('diagnosis', axis=1)
train_y = train['diagnosis']
test_X = test.drop('diagnosis', axis=1)
test_y = test['diagnosis']

# Z-Score标准化
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# 使用PCA将数据降维至2维
pca = PCA(n_components=2)
train_X_2d = pca.fit_transform(train_X)
test_X_2d = pca.transform(test_X)

# 定义参数范围
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

# 遍历每种参数组合并绘制决策边界
for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            # 如果使用线性核，不需要设置 gamma 参数
            if kernel == 'linear':
                model = svm.SVC(C=C, kernel=kernel)
            else:
                model = svm.SVC(C=C, gamma=gamma, kernel=kernel)

            # 训练模型
            model.fit(train_X_2d, train_y)

            # 在测试集上预测并计算准确率
            test_pred = model.predict(test_X_2d)
            accuracy = metrics.accuracy_score(test_y, test_pred)
            print(f"参数组合: C={C}, gamma={gamma}, kernel={kernel}, 测试集准确率={accuracy:.4f}")

            # 绘制决策边界
            plt.figure(figsize=(8, 6))
            x_min, x_max = train_X_2d[:, 0].min() - 1, train_X_2d[:, 0].max() + 1
            y_min, y_max = train_X_2d[:, 1].min() - 1, train_X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

            # 绘制训练样本点
            sns.scatterplot(x=train_X_2d[:, 0], y=train_X_2d[:, 1], hue=train_y,
                            palette=["blue", "red"], edgecolor='k')
            plt.title(f"SVM 决策边界 (C={C}, gamma={gamma}, kernel={kernel})\n测试集准确率: {accuracy:.4f}")
            plt.xlabel('主成分 1')
            plt.ylabel('主成分 2')
            plt.legend(['Benign', 'Malignant'])
            plt.show()
