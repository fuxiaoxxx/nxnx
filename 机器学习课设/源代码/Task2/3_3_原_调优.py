import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 加载数据
data = pd.read_csv('creditcard.csv')

# 数据预处理
X = data.drop('Class', axis=1)
y = data['Class']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义绘制ROC曲线的函数
def plot_roc_curve(fpr, tpr, roc_auc, label, color):
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})', color=color)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('接收者操作特征曲线 (ROC)')

# 模型列表
models = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# 定义颜色列表
colors = {
    'KNN': 'lightblue',
    'Naive Bayes': 'lightgreen',
    'Random Forest': 'orange'
}

# 超参数调优配置
param_grids = {
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Naive Bayes': {
        # 对于Naive Bayes，通常没有太多超参数可以调
        # 但我们可以选择使用不同的平滑方法（alpha）
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

# 不采样情况下的训练
for model_name, model in models.items():
    print(f"---- {model_name} 不采样 ----")

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳超参数
    best_model = grid_search.best_estimator_
    print(f"最佳超参数: {grid_search.best_params_}")

    # 训练模型并预测
    y_pred = best_model.predict(X_test_scaled)

    # 打印分类报告和混淆矩阵
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.title(f'{model_name} 不采样情况下混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} 不采样', color=colors[model_name])
    plt.legend(loc="lower right")
    plt.show()

    print(f"---- {model_name} 随机下采样 ----")

    # 使用RandomUnderSampler进行下采样
    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)

    # 使用最佳模型进行训练并预测
    best_model.fit(X_train_res, y_train_res)
    y_pred = best_model.predict(X_test_scaled)

    # 打印分类报告和混淆矩阵
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


    # 可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.title(f'{model_name} 随机下采样情况下混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} 随机下采样', color=colors[model_name])
    plt.legend(loc="lower right")
    plt.show()

    print(f"---- {model_name} SMOTE过采样 ----")

    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # 使用最佳模型进行训练并预测
    best_model.fit(X_train_res, y_train_res)
    y_pred = best_model.predict(X_test_scaled)

    # 打印分类报告和混淆矩阵
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print("混淆矩阵：")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.title(f'{model_name} SMOTE过采样情况下混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} SMOTE过采样', color=colors[model_name])
    plt.legend(loc="lower right")
    plt.show()
