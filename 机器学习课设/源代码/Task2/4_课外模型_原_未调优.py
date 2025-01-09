import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# 导入神经网络、XGBoost 和 LightGBM
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
plt.rcParams['font.sans-serif'] = 'SimHei'
# 加载数据
data = pd.read_csv('creditcard.csv')

# 数据预处理
# 特征和标签
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
    #plt.savefig(f'{label}_原_不调优')

# 模型列表：神经网络、XGBoost、LightGBM
models = {
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# 定义颜色列表
colors = {
    'Neural Network': 'lightblue',
    'XGBoost': 'lightgreen',
    'LightGBM': 'orange'
}

# 不采样情况下的训练
for model_name, model in models.items():
    print(f"---- {model_name} 不采样 ----")

    # 训练模型
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

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
    plt.savefig(f'{model_name}不采样_原_不调优_混淆矩阵')
    plt.close()  # 关闭图形，避免与后续图形冲突

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} 不采样', color=colors[model_name])

# 随机下采样情况下的训练
for model_name, model in models.items():
    print(f"---- {model_name} 随机下采样 ----")

    # 使用RandomUnderSampler进行下采样
    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)

    # 训练模型
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)

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
    plt.savefig(f'{model_name} 随机下采样_原_不调优_混淆矩阵')
    plt.close()  # 关闭图形，避免与后续图形冲突
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} 随机下采样', color=colors[model_name])

# SMOTE过采样情况下的训练
for model_name, model in models.items():
    print(f"---- {model_name} SMOTE过采样 ----")

    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # 训练模型
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)

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
    plt.savefig(f'{model_name} 过采样_原_不调优_混淆矩阵')
    plt.close()  # 关闭图形，避免与后续图形冲突

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, label=f'{model_name} SMOTE过采样', color=colors[model_name])

# 显示所有ROC曲线
plt.legend(loc="lower right")
plt.show()
