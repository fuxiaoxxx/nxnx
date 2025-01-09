# 数据
from matplotlib import pyplot as plt

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
values = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']
}
p_yes = {
    '色泽': [0.2, 0.3, 0.5],
    '根蒂': [0.1, 0.5, 0.4],
    '敲声': [0.3, 0.6, 0.1],
    '纹理': [0.7, 0.1, 0.2],
    '脐部': [0.5, 0.1, 0.4],
    '触感': [0.6667, 0.3333]
}
p_no = {
    '色泽': [0.4167, 0.3333, 0.25],
    '根蒂': [0.25, 0.3333, 0.4167],
    '敲声': [0.3333, 0.4167, 0.25],
    '纹理': [0.25, 0.3333, 0.4167],
    '脐部': [0.25, 0.4167, 0.3333],
    '触感': [0.6364, 0.3636]
}

# 绘图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    x = range(len(values[feature]))
    axes[row, col].bar(x, p_yes[feature], width=0.4, label='好瓜', color='lightgreen')
    axes[row, col].bar([i + 0.4 for i in x], p_no[feature], width=0.4, label='坏瓜', color='lightcoral')
    axes[row, col].set_xticks([i + 0.2 for i in x])
    axes[row, col].set_xticklabels(values[feature])
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('概率')
    axes[row, col].set_title(f'特征“{feature}”的概率分布')
    axes[row, col].legend()
plt.tight_layout()
plt.show()