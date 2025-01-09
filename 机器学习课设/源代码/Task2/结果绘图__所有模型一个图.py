import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['KNN', 'Bayes', 'Random Forest', 'Neural Network', 'XGBoost', 'LightGBM']
accuracy = [1.00, 0.98, 1.00, 1.00, 1.00, 1.00]
# 不采样数据（macro avg）
precision = [0.97, 0.53, 0.99, 0.92, 0.98, 0.61]
recall = [0.89, 0.90, 0.88, 0.89, 0.89, 0.80]
f1_score = [0.92, 0.55, 0.93, 0.90, 0.93, 0.66]

# 随机下采样数据
accuracy_rs = [0.98, 0.97, 0.98, 0.95, 0.96, 0.97]
# 随机下采样数据（macro avg）
precision_rs = [0.53, 0.53, 0.53, 0.52, 0.52, 0.52]
recall_rs = [0.94, 0.91, 0.96, 0.94, 0.95, 0.95]
f1_score_rs = [0.55, 0.54, 0.55, 0.52, 0.53, 0.53]

# SMOTE过采样数据
accuracy_smote = [1.00, 0.98, 1.00, 1.00, 1.00, 1.00]
# SMOTE过采样数据（macro avg）
precision_smote = [0.74, 0.53, 0.96, 0.89, 0.90, 0.78]
recall_smote = [0.93, 0.92, 0.92, 0.91, 0.92, 0.92]
f1_score_smote = [0.81, 0.55, 0.94, 0.90, 0.91, 0.84]


# 颜色定义
colors = {
    'accuracy': '#1f77b4',  # 蓝色
    'precision': '#ff7f0e',  # 橙色
    'recall': '#2ca02c',  # 绿色
    'f1_score': '#d62728',  # 红色
    'accuracy_rs': '#aec7e8',  # 浅蓝色
    'precision_rs': '#ffbb78',  # 浅橙色
    'recall_rs': '#98df8a',  # 浅绿色
    'f1_score_rs': '#ff9896',  # 浅红色
    'accuracy_smote': '#8c564b',  # 棕色
    'precision_smote': '#c49c94',  # 浅棕色
    'recall_smote': '#c5b0d5',  # 紫色
    'f1_score_smote': '#f7b6d2',  # 粉色
}

# 图1: 不采样下的直方图
x = np.arange(len(models))
width = 0.2  # 柱子的宽度

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy', color=colors['accuracy'])
rects2 = ax.bar(x - width/2, precision, width, label='Precision', color=colors['precision'])
rects3 = ax.bar(x + width/2, recall, width, label='Recall', color=colors['recall'])
rects4 = ax.bar(x + width*1.5, f1_score, width, label='F1-Score', color=colors['f1_score'])

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics (No Sampling)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# 图2: 随机下采样下的直方图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width*1.5, accuracy_rs, width, label='Accuracy', color=colors['accuracy_rs'])
rects2 = ax.bar(x - width/2, precision_rs, width, label='Precision', color=colors['precision_rs'])
rects3 = ax.bar(x + width/2, recall_rs, width, label='Recall', color=colors['recall_rs'])
rects4 = ax.bar(x + width*1.5, f1_score_rs, width, label='F1-Score', color=colors['f1_score_rs'])

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics (Random Downsampling)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# 图3: SMOTE过采样下的直方图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width*1.5, accuracy_smote, width, label='Accuracy', color=colors['accuracy_smote'])
rects2 = ax.bar(x - width/2, precision_smote, width, label='Precision', color=colors['precision_smote'])
rects3 = ax.bar(x + width/2, recall_smote, width, label='Recall', color=colors['recall_smote'])
rects4 = ax.bar(x + width*1.5, f1_score_smote, width, label='F1-Score', color=colors['f1_score_smote'])

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Metrics (SMOTE Oversampling)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# 图4: 各种模型不同采样下的指标比较
fig, ax = plt.subplots(figsize=(14, 8))
width = 0.2  # 柱子的宽度

# 绘制不同采样方式下的准确率
rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy (No Sampling)', color=colors['accuracy'])
rects2 = ax.bar(x - width/2, accuracy_rs, width, label='Accuracy (Random Downsampling)', color=colors['accuracy_rs'])
rects3 = ax.bar(x + width/2, accuracy_smote, width, label='Accuracy (SMOTE Oversampling)', color=colors['accuracy_smote'])

# 绘制不同采样方式下的精确度
rects4 = ax.bar(x - width*1.5, precision, width, label='Precision (No Sampling)', color=colors['precision'])
rects5 = ax.bar(x - width/2, precision_rs, width, label='Precision (Random Downsampling)', color=colors['precision_rs'])
rects6 = ax.bar(x + width/2, precision_smote, width, label='Precision (SMOTE Oversampling)', color=colors['precision_smote'])

# 绘制不同采样方式下的召回率
rects7 = ax.bar(x - width*1.5, recall, width, label='Recall (No Sampling)', color=colors['recall'])
rects8 = ax.bar(x - width/2, recall_rs, width, label='Recall (Random Downsampling)', color=colors['recall_rs'])
rects9 = ax.bar(x + width/2, recall_smote, width, label='Recall (SMOTE Oversampling)', color=colors['recall_smote'])

# 绘制不同采样方式下的F1分数
rects10 = ax.bar(x - width*1.5, f1_score, width, label='F1-Score (No Sampling)', color=colors['f1_score'])
rects11 = ax.bar(x - width/2, f1_score_rs, width, label='F1-Score (Random Downsampling)', color=colors['f1_score_rs'])
rects12 = ax.bar(x + width/2, f1_score_smote, width, label='F1-Score (SMOTE Oversampling)', color=colors['f1_score_smote'])

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparison of Performance Metrics Across Sampling Methods', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10)

plt.tight_layout()
plt.show()