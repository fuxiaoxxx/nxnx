import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
continuous_features = {
    '密度': {
        '是': {'平均值': 0.5561, '方差': 0.0142},
        '否': {'平均值': 0.4961, '方差': 0.0337}
    },
    '含糖率': {
        '是': {'平均值': 0.2529, '方差': 0.0048},
        '否': {'平均值': 0.1542, '方差': 0.0103}
    }
}

# 创建多子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 行 2 列

# 遍历每个连续型特征
for i, (feature, data) in enumerate(continuous_features.items()):
    # 提取均值和方差
    mean_yes = data['是']['平均值']
    var_yes = data['是']['方差']
    mean_no = data['否']['平均值']
    var_no = data['否']['方差']

    # 生成概率密度函数
    if feature == '密度':
        x = np.linspace(0.4, 0.7, 1000)  # 密度范围
    else:
        x = np.linspace(0.1, 0.35, 1000)  # 含糖率范围
    pdf_yes = norm.pdf(x, mean_yes, np.sqrt(var_yes))
    pdf_no = norm.pdf(x, mean_no, np.sqrt(var_no))

    # 绘制子图
    ax = axes[i]
    ax.plot(x, pdf_yes, label='好瓜', color='lightgreen')
    ax.plot(x, pdf_no, label='坏瓜', color='lightcoral')
    ax.set_xlabel(feature)
    ax.set_ylabel('概率密度')
    ax.set_title(f'特征“{feature}”的概率分布')
    ax.legend()

# 调整布局
plt.tight_layout()
plt.show()