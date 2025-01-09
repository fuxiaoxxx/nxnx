import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import 项目2_treePlotter
import 项目2_treeCreater

# 加载乳腺癌数据集
breast_cancer = datasets.load_breast_cancer()
X = pd.DataFrame(breast_cancer['data'], columns=breast_cancer['feature_names'])
y = pd.Series(breast_cancer['target_names'][breast_cancer['target']])

# 固定随机数种子，确保每次划分结果一致
random_state = 15

# 取三个样本为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 剩下样本中，取30个作为剪枝时的验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

# 定义保存结果的函数
def save_results(filename, results):
    with open(filename, 'w') as f:
        for method, accuracy in results.items():
            f.write(f'{method}:{accuracy}\n')

# 不剪枝
tree_no_pruning = 项目2_treeCreater.DecisionTree('gini')
tree_no_pruning.fit(X_train, y_train, X_val, y_val)
no_pruning_accuracy = np.mean(tree_no_pruning.predict(X_test) == y_test)
print('不剪枝：', no_pruning_accuracy)
项目2_treePlotter.create_plot(tree_no_pruning.tree_)

# 预剪枝
tree_pre_pruning = 项目2_treeCreater.DecisionTree('gini', 'pre_pruning')
tree_pre_pruning.fit(X_train, y_train, X_val, y_val)
pre_pruning_accuracy = np.mean(tree_pre_pruning.predict(X_test) == y_test)
print('预剪枝：', pre_pruning_accuracy)
项目2_treePlotter.create_plot(tree_pre_pruning.tree_)

# 后剪枝
tree_post_pruning = 项目2_treeCreater.DecisionTree('gini', 'post_pruning')
tree_post_pruning.fit(X_train, y_train, X_val, y_val)
post_pruning_accuracy = np.mean(tree_post_pruning.predict(X_test) == y_test)
print('后剪枝：', post_pruning_accuracy)
项目2_treePlotter.create_plot(tree_post_pruning.tree_)

# 保存结果到txt文件
results = {
    '不剪枝': no_pruning_accuracy,
    '预剪枝': pre_pruning_accuracy,
    '后剪枝': post_pruning_accuracy
}
filename = 'gini-breast_cancer.txt'
save_results(filename, results)