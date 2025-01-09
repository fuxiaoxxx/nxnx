import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 加载数据
data = pd.read_csv('creditcard.csv')

# 2. 数据预处理
# 特征和标签
X = data.drop('Class', axis=1)  # 去掉标签列
y = data['Class']  # 标签

# 标准化数据（PCA对标准化后的数据更敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 应用PCA降维
# 这里不设置n_components，计算所有的主成分
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. 查看每个主成分解释的方差比例
print("每个主成分的方差解释比例：")
print(pca.explained_variance_ratio_)

# 5. 计算累计的方差比例，帮助决定选择主成分的个数
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
print("累计方差解释比例：")
print(cumulative_variance_ratio)

# 6. 可视化累计方差比例，帮助判断保留主成分的数量
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance_ratio, marker='o', linestyle='--', color='b')
plt.title('累计方差解释比例')
plt.xlabel('主成分数量')
plt.ylabel('累计方差比例')
plt.grid(True)
plt.show()

# 7. 根据累计方差比例选择合适的主成分数量（例如，保留95%的方差）
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"选择保留前 {n_components} 个主成分以解释95%的方差。")

# 8. 根据选定的主成分数量重新应用PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# 9. 将降维后的数据与标签合并
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
data_pca = pd.concat([X_pca_df, y], axis=1)

# 10. 保存处理后的数据到新的CSV文件
data_pca.to_csv('creditcard_pca.csv', index=False)

print("PCA数据处理完成，已保存为 'creditcard_pca.csv'")
