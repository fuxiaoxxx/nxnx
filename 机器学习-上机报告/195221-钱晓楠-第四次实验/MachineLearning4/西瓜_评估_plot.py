import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 数据
data = [
    {"C": 1, "gamma": 0.1, "kernel": "linear", "accuracy": 0.82},
    {"C": 1, "gamma": 0.1, "kernel": "rbf", "accuracy": 0.76},
    {"C": 1, "gamma": 1, "kernel": "linear", "accuracy": 0.82},
    {"C": 1, "gamma": 1, "kernel": "rbf", "accuracy": 1.00},
    {"C": 10, "gamma": 0.1, "kernel": "linear", "accuracy": 0.82},
    {"C": 10, "gamma": 0.1, "kernel": "rbf", "accuracy": 0.88},
    {"C": 10, "gamma": 1, "kernel": "linear", "accuracy": 0.82},
    {"C": 10, "gamma": 1, "kernel": "rbf", "accuracy": 1.00},
    {"C": 100, "gamma": 0.1, "kernel": "linear", "accuracy": 0.82},
    {"C": 100, "gamma": 0.1, "kernel": "rbf", "accuracy": 1.00},
    {"C": 100, "gamma": 1, "kernel": "linear", "accuracy": 0.82},
    {"C": 100, "gamma": 1, "kernel": "rbf", "accuracy": 1.00}
]

# 创建数据框
df = pd.DataFrame(data)

# 按 kernel 类型分组
linear_df = df[df['kernel'] == 'linear']
rbf_df = df[df['kernel'] == 'rbf']

# 创建热力图数据
linear_heatmap_data = pd.pivot_table(linear_df, values='accuracy', index='gamma', columns='C')
rbf_heatmap_data = pd.pivot_table(rbf_df, values='accuracy', index='gamma', columns='C')

# 绘制热力图
plt.figure(figsize=(20, 6))

# Linear 核的热力图
plt.subplot(1, 2, 1)
sns.heatmap(linear_heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
plt.title('Linear Kernel')
plt.xlabel('C')
plt.ylabel('gamma')

# RBF 核的热力图
plt.subplot(1, 2, 2)
sns.heatmap(rbf_heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
plt.title('RBF Kernel')
plt.xlabel('C')
plt.ylabel('gamma')

# 显示图表
plt.tight_layout()
plt.show()

'''折线图'''
# 创建数据框
df = pd.DataFrame(data)

# 按 kernel 类型分组
linear_df = df[df['kernel'] == 'linear']
rbf_df = df[df['kernel'] == 'rbf']

# 绘制折线图
plt.figure(figsize=(12, 6))

# Linear 核的折线图
plt.subplot(1, 2, 1)
for gamma in linear_df['gamma'].unique():
    subset = linear_df[linear_df['gamma'] == gamma]
    plt.plot(subset['C'], subset['accuracy'], marker='o', label=f'gamma={gamma}')
plt.title('Linear Kernel')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()

# RBF 核的折线图
plt.subplot(1, 2, 2)
for gamma in rbf_df['gamma'].unique():
    subset = rbf_df[rbf_df['gamma'] == gamma]
    plt.plot(subset['C'], subset['accuracy'], marker='o', label=f'gamma={gamma}')
plt.title('RBF Kernel')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()