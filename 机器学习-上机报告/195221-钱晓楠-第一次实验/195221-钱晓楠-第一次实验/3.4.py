import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
bc_data = pd.read_csv(r"dataR2.csv", skiprows=1)
pima_data = pd.read_csv(r"diabetes.csv",skiprows=1)
def preprocess_data(data):
    dataSet = data.values  # pd.DataFrame to np.array
    dataArr = dataSet[:, :-1]  # feature array
    labelArr = dataSet[:, -1].reshape(-1, 1)  # label array

    # 标准化
    dataMax = dataArr.max(axis=0)
    dataMin = dataArr.min(axis=0)
    dataArr = (dataArr - dataMax) / (dataMax - dataMin)

    # 增加偏置参数
    dataArr = np.hstack((dataArr, np.ones([dataArr.shape[0], 1])))
    # 因为原来数据中1代表健康, 2代表病人，用逻辑回归要将2变为0
    labelArr[labelArr == 2] = 0
    return dataArr, labelArr

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logit_regression(theta, x, y, iteration=1000, learning_rate=0.001, lbd=0.01):
    # 获取样本数量
    m = y.shape[0]

    # 确保 theta 的形状与 x 的列数匹配
    if theta.shape[0] != x.shape[1]:
        theta = np.random.rand(x.shape[1], 1)

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

def predict(theta, x):
    linear_combination = np.dot(x, theta)  # 计算线性组合
    y_prob = sigmoid(linear_combination)  # 计算sigmoid值

    # 根据阈值进行预测
    pre = (y_prob >= 0.5).astype(int)  # 将布尔值转换为整数
    return pre.flatten()  # 直接返回一维数组

def cross_validation(data):
    x, y = preprocess_data(data)
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    error_rates = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 随机初始化theta
        theta = np.random.rand(x_train.shape[1], 1)
        theta = logit_regression(theta, x_train, y_train)

        # 预测
        predictions = predict(theta, x_test)

        # 计算错误率
        error_rate = 1 - accuracy_score(y_test, predictions)
        error_rates.append(error_rate)

        # 打印分类报告
        print(classification_report(y_test, predictions, target_names=['健康', '患病'], zero_division=0))

    # 计算平均错误率
    avg_error_rate = np.mean(error_rates)
    print(f"平均错误率: {avg_error_rate:.4f}")

# 运行交叉验证
print("Breast Cancer Coimbra 数据集:")
cross_validation(bc_data)
print("Pima Indians Diabetes Database 数据集：")
cross_validation(pima_data)