import numpy as np  # 用于矩阵计算
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

'''注意pd读入数据为dataframe类型，要转化为array类型，机器学习中的输入数据均为array类型'''

data = pd.read_csv('D:/学习资料/机器学习/西瓜数据集3.0alpha.txt', sep=' ')
dataSet = data.values

# 分离数据和目标属性
X = dataSet[:, :-1]  # 特征矩阵
print(X)
y = dataSet[:, -1]  # 标签向量

'''自编码实现LDA'''
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
print("X的形状：")
print(X_test.shape)
# 1-st. 获取每个类别的均值向量
u = []
for i in range(2):  # 两个类别
    u.append(np.mean(X_train[y_train == i], axis=0))  # 使用训练集的均值

# 2-nd. 计算类内散布矩阵
m, n = np.shape(X_train)
Sw = np.zeros((n, n))
for i in range(m):
    x_tmp = X_train[i].reshape(n, 1)  # 行 -> 列向量
    u_tmp = u[0].reshape(n, 1) if y_train[i] == 0 else u[1].reshape(n, 1)
    Sw += np.dot(x_tmp - u_tmp, (x_tmp - u_tmp).T)

Sw = np.mat(Sw)
U, sigma, V = np.linalg.svd(Sw)

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
# 3-th. 计算参数w
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))
print("w:")
print(w)
# 4-th. 在散点图中绘制LDA线
f3 = plt.figure(3)
plt.xlim(-0.2, 1)
plt.ylim(-0.5, 0.7)

p0_x0 = -X_train[:, 0].max()
p0_x1 = (w[1, 0] / w[0, 0]) * p0_x0
p1_x0 = X_train[:, 0].max()
p1_x1 = (w[1, 0] / w[0, 0]) * p1_x0

plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker='o', color='g', s=10, label='good')
plt.legend(loc='upper right')
plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])

# 预测测试集
y_pred = np.dot(X_test, w)  # 这将返回 (5, 1)

# 根据阈值进行分类，并保持形状为 (5, 1)
y_pred_classes = (y_pred > 0).astype(int)  # 保持形状为 (5, 1)

# 将 y_test 和 y_pred_classes 转换为 NumPy 数组
y_test = np.asarray(y_test).reshape(-1, 1)  # 确保 y_test 是列向量
y_pred_classes = np.asarray(y_pred_classes)  # 转换为数组

# 评估结果
accuracy = accuracy_score(y_test.flatten(), y_pred_classes.flatten())  # 展平以确保与 y_test 一致
f1 = f1_score(y_test.flatten(), y_pred_classes.flatten())

# 生成分类报告
report = classification_report(y_test.flatten(), y_pred_classes.flatten(), target_names=['0.0', '1.0'])

print(f'测试集准确率: {accuracy:.2f}')
print(report)
# 在线上绘制投影点

def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if   k == 0:      return [a, t]
    elif k == np.inf: return [0, b]
    x = (a+k*b-k*t) / (k*k+1)
    y = k*x + t
    return [x, y]

m, n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
    if y[i] == 0:
        plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
    if y[i] == 1:
        plt.plot(x_p[0], x_p[1], 'go', markersize=5)
    plt.plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)

plt.show()

'''使用sklearn进行LDA'''
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)

# 模型拟合
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)

# 模型验证
y_pred = lda_model.predict(X_test)

# 总结模型的拟合效果
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# 绘制分类器的决策边界
f2 = plt.figure(2)
h = 0.001
x0, x1 = np.meshgrid(np.arange(-1, 1, h),
                     np.arange(-1, 1, h))

z = lda_model.predict(np.c_[x0.ravel(), x1.ravel()])

# 将结果放入颜色图中
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, alpha=0.3)

# 绘制训练点
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')

# 计算LDA模型的决策边界
w = lda_model.coef_[0]
b = lda_model.intercept_[0]

# 计算决策边界的斜率和截距
slope = -w[0] / w[1]
intercept = -b / w[1]

# 绘制决策边界
x_vals = np.array(plt.xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, 'r--', label='Decision Boundary')

plt.legend(loc='upper right')
plt.show()

