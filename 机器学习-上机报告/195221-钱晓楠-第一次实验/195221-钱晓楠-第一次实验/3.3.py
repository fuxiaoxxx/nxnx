import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# 数据加载
data = pd.read_csv('D:/学习资料/机器学习/西瓜数据集3.0alpha.txt', sep=' ')

x = data[['密度', '含糖率']].values

'''提取数据方式1'''
# 提取特征和标签
density = data['密度'].values.reshape(-1, 1)
sugar_rate = data['含糖率'].values.reshape(-1, 1)
ytrain = data['好瓜'].values.reshape(-1, 1)
print('ytrain:')
print(ytrain.shape)

# 原始特征矩阵
xtrain = np.hstack((density, sugar_rate))

# 添加偏置项到特征矩阵的右边
xtrain = np.hstack((xtrain, np.ones([density.shape[0], 1])))
print(xtrain.shape)
# 留出法，训练数据和测试数据划分
xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.25, random_state=33)

'''提取数据方式2'''
dataSet = data.values  # pd.DataFrame to np.array 注意要转换为array类型
dataArr = dataSet[:, :-1]  # feature array
labelArr = dataSet[:, -1].reshape(-1,1)  # label array
# 增加偏置参数
dataArr = np.hstack((dataArr, np.ones([dataArr.shape[0], 1])))
print(labelArr.shape)
print(dataArr.shape)
xtrain, xtest, ytrain, ytest = train_test_split(dataArr, labelArr, test_size=0.25, random_state=33)


'''math解法'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logit_regression(theta, x, y, iteration=1000, learning_rate=0.01, lbd=0.01):
    # 获取样本数量
    m = y.shape[0]

    # 迭代更新参数
    for i in range(iteration):
        # 计算线性组合
        linear_model = np.dot(x, theta)

        # 计算预测值
        y_predicted = sigmoid(linear_model)

        # 计算梯度
        gradient = np.dot(x.transpose(), (y_predicted - y)) / m

        # 更新参数
        theta = theta - learning_rate * (gradient + lbd * theta)

        # 计算损失函数
        cost = -1 / m * (np.dot(y.transpose(), np.log(y_predicted)) + np.dot((1 - y).transpose(),
                                                                             np.log(1 - y_predicted))) + lbd / (2 * m) * np.dot(theta.transpose(), theta)
    return theta

# def predict(theta, x):
#     pre = np.zeros([x.shape[0], 1])
#     for idx, valu in enumerate(np.dot(x, theta)):
#         if sigmoid(valu) >= 0.5:
#             pre[idx] = 1
#         else:
#             pre[idx] = 0
#     return pre
def predict(theta, x):
    linear_combination = np.dot(x, theta)  # 计算线性组合
    y_prob = sigmoid(linear_combination)  # 计算sigmoid值

    # 根据阈值进行预测
    pre = (y_prob >= 0.5).astype(int)  # 将布尔值转换为整数
    return pre.flatten()  # 直接返回一维数组

#随机初始化theta,为参数矩阵，包括w1,w2和b
theta= np.random.rand(3, 1)
theta = logit_regression(theta, xtrain, ytrain, learning_rate=1)
pre = predict(theta, xtest)
print('predictions are\n', pre)
print('ground truth is', ytest)
print('theta is ', theta)
print('the accuracy is', np.mean(pre == ytest))
print(classification_report(ytest, pre, target_names=['0', '1']))

# 绘制训练集的所有点
plt.figure(figsize=(10, 6))
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, cmap='brg', s=80, label='Training Data')

# 绘制测试集的预测结果
plt.scatter(xtest[:, 0], xtest[:, 1], c=pre, cmap='brg', s=80, marker='x', label='Test Predictions')

# 绘制逻辑回归的决策边界
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 计算网格上的预测值
Z = predict(theta, np.c_[xx.ravel(), yy.ravel(),np.ones(xx.ravel().shape[0])])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

plt.title('Watermelon Classification')
plt.xlabel('Density', fontsize=14)
plt.ylabel('Sugar Content', fontsize=14)
plt.tick_params(labelsize=10)
plt.legend()
plt.show()

'''sklearn'''

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as sl
import sklearn.metrics as sm

x_train=xtrain[:, :-1]
y_train=ytrain
y_test = ytest
x_test=xtest[:, :-1]

# 创建模型
model = sl.LogisticRegression()

'''没有正则化的模型'''
# 模型训练
model.fit(x_train, y_train)  # 忽略偏置项
pred_y = model.predict(x_test)
acc = model.score(x_train, ytrain.ravel())
print('score', acc)

# 模型评估
print('精度：', sm.accuracy_score(ytest, pred_y))
print('查准率：', sm.precision_score(ytest, pred_y, average='macro'))
print('召回率：', sm.recall_score(ytest, pred_y, average='macro'))
print('f1得分：', sm.f1_score(ytest, pred_y, average='macro'))
print('report', sm.classification_report(ytest, pred_y))

'''含有正则化的模型'''
# 创建模型，C值越小，代表正则化越强
model = sl.LogisticRegression(solver='sag',C=200)
model.fit(x_train, y_train)
pred_y = model.predict(x_test)
acc = model.score(x_train, y_train)
print('score', acc)

# 分类任务性能度量：精度（accuracy）、查准率（precision）、查全率（recall，召回率）和F1
print('精度：', sm.accuracy_score(y_test, pred_y))
print('查准率：', sm.precision_score(y_test, pred_y, average='macro'))
print('召回率：', sm.recall_score(y_test, pred_y, average='macro'))
print('f1得分：', sm.f1_score(y_test, pred_y, average='macro'))
print('report', sm.classification_report(y_test, pred_y))

# 绘制训练集的所有点
plt.figure(figsize=(10, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='brg', s=80, label='Training Data')

# 绘制测试集的预测结果
plt.scatter(x_test[:, 0], x_test[:, 1], c=pred_y, cmap='brg', s=80, marker='x', label='Test Predictions')

# 绘制逻辑回归的决策边界
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='brg')

plt.title('Watermelon Classification')
plt.xlabel('Density', fontsize=14)
plt.ylabel('Sugar Content', fontsize=14)
plt.tick_params(labelsize=10)
plt.legend()
plt.show()
