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

# 指标和对应的数据
metrics = ['Precision', 'Sensitivity', 'Accuracy', 'F1 Score']
x = np.arange(len(metrics))  # x轴的位置
width = 0.25  # 柱子的宽度

# 创建一个函数来绘制每个模型的柱状图
def plot_model_results(model_index, model_name):
    # 提取数据
    model_no_sampling = [precision[model_index], recall[model_index], accuracy[model_index], f1_score[model_index]]
    model_rs = [precision_rs[model_index], recall_rs[model_index], accuracy_rs[model_index], f1_score_rs[model_index]]
    model_smote = [precision_smote[model_index], recall_smote[model_index], accuracy_smote[model_index], f1_score_smote[model_index]]

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, model_no_sampling, width, label='No Sampling', color='skyblue')
    rects2 = ax.bar(x, model_rs, width, label='Random Undersampling', color='lightcoral')
    rects3 = ax.bar(x + width, model_smote, width, label='SMOTE', color='goldenrod')

    # 添加标题和标签
    ax.set_title(f'{model_name} Results', fontsize=18, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)

    # 显示数值
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    filename = f'{model_name}不同采样方式直方图.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
    plt.close()  # 关闭当前图表，避免内存泄漏

# 为每个模型绘制图表
for i, model_name in enumerate(models):
    plot_model_results(i, model_name)