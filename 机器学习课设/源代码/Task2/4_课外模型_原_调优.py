import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# 导入神经网络、XGBoost 和 LightGBM
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

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


# 模型列表：神经网络、XGBoost、LightGBM
models = {
    'Neural Network': MLPClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# 超参数搜索空间
param_grids = {
    'Neural Network': {
        'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],  # 调节隐藏层的大小
        'activation': ['relu', 'tanh'],  # 激活函数
        'solver': ['adam', 'sgd'],  # 求解器
        'learning_rate': ['constant', 'adaptive'],  # 学习率策略
        'max_iter': [500, 1000]  # 最大迭代次数
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [3, 5, 7],  # 树的最大深度
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率
        'subsample': [0.8, 1.0],  # 样本子集比例
        'colsample_bytree': [0.8, 1.0]  # 特征子集比例
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [3, 5, 7],  # 树的最大深度
        'learning_rate': [0.01, 0.05, 0.1],  # 学习率
        'num_leaves': [31, 50, 100],  # 每棵树的叶子节点数
        'subsample': [0.8, 1.0]  # 样本子集比例
    }
}

# 定义颜色列表
colors = {
    'Neural Network': 'lightblue',
    'XGBoost': 'lightgreen',
    'LightGBM': 'orange'
}

# 结果保存
results = {
    'model': [],
    'sampling': [],
    'classification_report': [],
    'confusion_matrix': [],
    'roc_auc': [],
    'roc_curve': []
}


# 训练模型并记录结果
def train_and_evaluate(model_name, model, X_train_scaled, y_train, X_test_scaled, y_test, sampling_strategy=None):
    # 采样策略
    if sampling_strategy == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)
    elif sampling_strategy == 'smote':
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_res, y_train_res)

    # 获取最佳超参数
    best_model = grid_search.best_estimator_
    print(f"{model_name} - 最佳超参数: {grid_search.best_params_}")

    # 训练模型并进行预测
    y_pred = best_model.predict(X_test_scaled)

    # 记录分类报告
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    # 记录混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    roc_auc_value = auc(fpr, tpr)

    # 记录结果
    results['model'].append(model_name)
    results['sampling'].append(sampling_strategy if sampling_strategy else 'no sampling')
    results['classification_report'].append(classification_rep)
    results['confusion_matrix'].append(cm)
    results['roc_auc'].append(roc_auc_value)
    results['roc_curve'].append((fpr, tpr, roc_auc_value))

    return best_model, classification_rep, cm, fpr, tpr, roc_auc_value


# 训练模型并收集每种采样方法的结果
for model_name, model in models.items():
    for sampling_strategy in ['no sampling', 'undersample', 'smote']:
        print(f"---- {model_name} 使用 {sampling_strategy} ----")
        best_model, classification_rep, cm, fpr, tpr, roc_auc_value = train_and_evaluate(
            model_name, model, X_train_scaled, y_train, X_test_scaled, y_test, sampling_strategy
        )

        # 打印分类报告和混淆矩阵
        print(f"分类报告：\n{classification_rep}")
        print(f"混淆矩阵：\n{cm}")

        # 可视化混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"])
        plt.title(f'{model_name} - {sampling_strategy} - 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()

        # 绘制ROC曲线
        plot_roc_curve(fpr, tpr, roc_auc_value, label=f'{model_name} ({sampling_strategy})', color=colors[model_name])

# 显示所有ROC曲线
plt.legend(loc="lower right")
plt.show()

# 最终汇总所有模型的调优结果
for i in range(len(results['model'])):
    print(f"模型: {results['model'][i]}")
    print(f"采样策略: {results['sampling'][i]}")
    print(f"AUC: {results['roc_auc'][i]:.2f}")
    print(f"分类报告: {results['classification_report'][i]}")
    print(f"混淆矩阵: \n{results['confusion_matrix'][i]}")
    print("-" * 80)
