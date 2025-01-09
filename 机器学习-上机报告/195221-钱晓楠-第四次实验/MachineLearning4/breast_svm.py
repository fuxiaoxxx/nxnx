# -*- coding: utf-8 -*-
# 乳腺癌诊断分类
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("./data.csv")

# 数据探索
pd.set_option('display.max_columns', None)
print(data['diagnosis'].value_counts())

# 数据清洗
data.drop("id", axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
print(data['diagnosis'].value_counts())

# 可视化肿瘤诊断结果
plt.figure(figsize=(8, 6))
sns.countplot(x='diagnosis', data=data, label="Count")
plt.xlabel('Diagnosis (0: Benign, 1: Malignant)')
plt.ylabel('Count')
plt.title('Tumor Diagnosis Count')
plt.xticks(ticks=[0, 1], labels=['Benign', 'Malignant'])
plt.legend()
plt.show()

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

# 定义参数范围
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

# 手动搜索最佳参数组合
best_score = 0
best_params = {}

for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            # 如果使用线性核，不需要设置 gamma 参数
            if kernel == 'linear':
                model = svm.SVC(C=C, kernel=kernel)
            else:
                model = svm.SVC(C=C, gamma=gamma, kernel=kernel)

            # 训练模型
            model.fit(train_X, train_y)

            # 在测试集上评估
            prediction = model.predict(test_X)
            accuracy = metrics.accuracy_score(test_y, prediction)
            print(f"参数组合: C={C}, gamma={gamma}, kernel={kernel}, 准确率={accuracy:.4f}")

            # 保存最佳参数组合
            if accuracy > best_score:
                best_score = accuracy
                best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}

print("\n最佳参数组合:", best_params)
print("最佳测试集准确率:", best_score)
