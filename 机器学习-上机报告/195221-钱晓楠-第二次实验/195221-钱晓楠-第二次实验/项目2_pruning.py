import pandas as pd
import numpy as np


def post_pruning(X_train, y_train, X_val, y_val, tree_=None):
    """
    后剪枝函数，用于对决策树进行后剪枝操作。

    参数:
    X_train: 训练集特征数据，pd.DataFrame 类型。
    y_train: 训练集标签数据，pd.Series 类型。
    X_val: 验证集特征数据，pd.DataFrame 类型。
    y_val: 验证集标签数据，pd.Series 类型。
    tree_: 当前节点，Node 类型。

    返回:
    剪枝后的决策树节点。
    """
    # 如果当前节点已经是叶子节点，直接返回
    if tree_.is_leaf:
        return tree_

    # 如果验证集为空，不再剪枝，直接返回
    if X_val.empty:
        return tree_

    # 计算训练集中样本最多的类别
    most_common_in_train = pd.value_counts(y_train).index[0]
    # 计算当前节点下验证集样本的准确率
    current_accuracy = np.mean(y_val == most_common_in_train)

    # 如果当前节点是连续值特征
    if tree_.is_continuous:
        # 根据分割点将训练集和验证集分为两部分
        up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
        up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

        # 递归处理上部分子树
        up_subtree = post_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                  y_val[up_part_val],
                                  tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
        tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree

        # 递归处理下部分子树
        down_subtree = post_pruning(X_train[down_part_train], y_train[down_part_train],
                                    X_val[down_part_val], y_val[down_part_val],
                                    tree_.subtree['< {:.3f}'.format(tree_.split_value)])
        tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree

        # 更新当前节点的高度和叶子节点数量
        tree_.high = max(up_subtree.high, down_subtree.high) + 1
        tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

        # 如果上部分和下部分子树都是叶子节点
        if up_subtree.is_leaf and down_subtree.is_leaf:
            # 定义分割函数
            def split_fun(x):
                if x >= tree_.split_value:
                    return '>= {:.3f}'.format(tree_.split_value)
                else:
                    return '< {:.3f}'.format(tree_.split_value)

            # 根据分割函数对验证集进行分割
            val_split = X_val.loc[:, tree_.feature_name].map(split_fun)
            # 计算分割后的验证集准确率
            right_class_in_val = y_val.groupby(val_split).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            # 如果当前节点为叶子节点时的准确率大于不剪枝的准确率，则进行剪枝操作
            if current_accuracy > split_accuracy:
                set_leaf(pd.value_counts(y_train).index[0], tree_)
    else:
        # 初始化最大高度和叶子节点数量
        max_high = -1
        tree_.leaf_num = 0
        is_all_leaf = True  # 判断当前节点下，所有子树是否都为叶节点

        # 遍历当前节点的所有子树
        for key in tree_.subtree.keys():
            # 根据特征值将训练集和验证集分为两部分
            this_part_train = X_train.loc[:, tree_.feature_name] == key
            this_part_val = X_val.loc[:, tree_.feature_name] == key

            # 递归处理子树
            tree_.subtree[key] = post_pruning(X_train[this_part_train], y_train[this_part_train],
                                              X_val[this_part_val], y_val[this_part_val], tree_.subtree[key])
            # 更新最大高度和叶子节点数量
            if tree_.subtree[key].high > max_high:
                max_high = tree_.subtree[key].high
            tree_.leaf_num += tree_.subtree[key].leaf_num

            # 判断子树是否为叶子节点
            if not tree_.subtree[key].is_leaf:
                is_all_leaf = False
        # 更新当前节点的高度
        tree_.high = max_high + 1

        # 如果所有子节点都为叶子节点，则考虑是否进行剪枝
        if is_all_leaf:
            # 计算分割后的验证集准确率
            right_class_in_val = y_val.groupby(X_val.loc[:, tree_.feature_name]).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            # 如果当前节点为叶子节点时的准确率大于不剪枝的准确率，则进行剪枝操作
            if current_accuracy > split_accuracy:
                set_leaf(pd.value_counts(y_train).index[0], tree_)

    # 返回剪枝后的节点
    return tree_


def pre_pruning(X_train, y_train, X_val, y_val, tree_=None):
    # 如果当前节点已经是叶节点，直接返回
    if tree_.is_leaf:
        return tree_

    # 如果验证集为空，则不进行剪枝，直接返回当前树
    if X_val.empty:
        return tree_

    # 计算训练集中样本最多的类别
    most_common_in_train = pd.value_counts(y_train).index[0]
    # 计算当前节点在验证集上的准确率
    current_accuracy = np.mean(y_val == most_common_in_train)

    # 如果当前节点是连续特征
    if tree_.is_continuous:
        # 计算分割后的准确率
        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val,
                                                  split_value=tree_.split_value)

        # 比较当前准确率和分割后的准确率
        if current_accuracy >= split_accuracy:
            # 如果当前准确率更高，则将当前节点设置为叶节点
            set_leaf(pd.value_counts(y_train).index[0], tree_)
        else:
            # 根据分割值将训练集和验证集分为上下两部分
            up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
            up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

            # 递归处理上部分子树
            up_subtree = pre_pruning(X_train[up_part_train], y_train[up_part_train],
                                     X_val[up_part_val], y_val[up_part_val],
                                     tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
            tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree

            # 递归处理下部分子树
            down_subtree = pre_pruning(X_train[down_part_train], y_train[down_part_train],
                                       X_val[down_part_val], y_val[down_part_val],
                                       tree_.subtree['< {:.3f}'.format(tree_.split_value)])
            tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree

            # 更新当前节点的高度和叶子节点数量
            tree_.high = max(up_subtree.high, down_subtree.high) + 1
            tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

    else:  # 如果是离散特征
        # 计算分割后的准确率
        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val)

        # 比较当前准确率和分割后的准确率
        if current_accuracy >= split_accuracy:
            # 如果当前准确率更高，则将当前节点设置为叶节点
            set_leaf(pd.value_counts(y_train).index[0], tree_)
        else:
            max_high = -1
            tree_.leaf_num = 0
            # 遍历每个子树
            for key in tree_.subtree.keys():
                # 根据特征值将训练集和验证集分为子部分
                this_part_train = X_train.loc[:, tree_.feature_name] == key
                this_part_val = X_val.loc[:, tree_.feature_name] == key

                # 递归处理子树
                tree_.subtree[key] = pre_pruning(X_train[this_part_train], y_train[this_part_train],
                                                 X_val[this_part_val], y_val[this_part_val],
                                                 tree_.subtree[key])

                # 更新最大高度和叶子节点数量
                if tree_.subtree[key].high > max_high:
                    max_high = tree_.subtree[key].high
                tree_.leaf_num += tree_.subtree[key].leaf_num

            # 更新当前节点的高度
            tree_.high = max_high + 1

    # 返回剪枝后的节点
    return tree_


def set_leaf(leaf_class, tree_):
    # 设置节点为叶节点
    tree_.is_leaf = True  # 若划分前正确率大于划分后正确率。则选择不划分，将当前节点设置为叶节点
    tree_.leaf_class = leaf_class
    tree_.feature_name = None
    tree_.feature_index = None
    tree_.subtree = {}
    tree_.impurity = None
    tree_.split_value = None
    tree_.high = 0  # 重新设立高 和叶节点数量
    tree_.leaf_num = 1


def val_accuracy_after_split(feature_train, y_train, feature_val, y_val, split_value=None):
    # 若是连续值时，需要需要按切分点对feature 进行分组，若是离散值，则不用处理
    if split_value is not None:
        def split_fun(x):
            if x >= split_value:
                return '>= {:.3f}'.format(split_value)
            else:
                return '< {:.3f}'.format(split_value)

        train_split = feature_train.map(split_fun)
        val_split = feature_val.map(split_fun)

    else:
        train_split = feature_train
        val_split = feature_val

    majority_class_in_train = y_train.groupby(train_split).apply(
        lambda x: pd.value_counts(x).index[0])  # 计算各特征下样本最多的类别
    right_class_in_val = y_val.groupby(val_split).apply(
        lambda x: np.sum(x == majority_class_in_train[x.name]))  # 计算各类别对应的数量

    return right_class_in_val.sum() / y_val.shape[0]  # 返回准确率

