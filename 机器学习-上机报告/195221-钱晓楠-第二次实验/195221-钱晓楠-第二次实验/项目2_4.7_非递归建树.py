from collections import deque
'''
拟合函数中，X支持pd.DataFrame数据类型；y暂只支持pd.Series类型，其他数据类型未测试，
目前在西瓜数据集上和sklearn中自带的iris数据集上运行正常，以后若发现有其他bug，再修复。
'''
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import deque
#用于表示决策树中的一个节点
class Node(object):
    def __init__(self):
        self.feature_name = None
        self.feature_index = None
        self.subtree = {}
        self.impurity = None
        self.is_continuous = False
        self.split_value = None
        self.is_leaf = False
        self.leaf_class = None
        self.leaf_num = 0
        self.high = -1
'''
feature_name: 当前节点用于划分的特征名称。
feature_index: 当前节点用于划分的特征索引。
subtree: 当前节点的子树，字典类型，键为特征值或分割点，值为子节点。
impurity: 当前节点的杂质度量（如基尼系数、信息增益等）。
is_continuous: 当前节点特征是否为连续值。
split_value: 当前节点特征为连续值时的分割点。
is_leaf: 当前节点是否为叶子节点。
leafclass: 当前叶子节点的类别，也就是标签值。
leaf_num: 当前节点及其子树中叶子节点的总数。
high: 当前节点的高度（深度）。
'''
class DecisionTreeNonRecursive:
    def __init__(self, criterion='gini', pruning=None, MaxDepth=5):
        assert criterion in ('gini', 'infogain', 'gainratio')
        assert pruning in (None, 'pre_pruning', 'post_pruning')
        self.criterion = criterion
        self.pruning = pruning
        self.MaxDepth = MaxDepth
        print(f"选择的划分方法为：{self.criterion}, 最大深度为：{self.MaxDepth}")

    def fit(self, X_train, y_train):
        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        self.columns = list(X_train.columns)
        self.tree_ = self.generate_tree(X_train, y_train)
        return self

    def generate_tree(self, X, y):
        root = Node()
        root.high = 0  # 根节点的高度为0
        queue = deque([(root, X, y)])  # 使用队列来存储节点和数据
        nodes_to_process = []  # 记录所有节点以便后续计算 leaf_num

        while queue:
            current_node, current_X, current_y = queue.popleft()
            nodes_to_process.append(current_node)

            # 叶节点条件：达到最大深度或只有单一类别或没有特征
            if current_node.high >= self.MaxDepth or current_y.nunique() == 1 or current_X.empty:
                current_node.is_leaf = True
                current_node.leaf_class = current_y.mode()[0]
                current_node.leaf_num = 1  # 是叶子节点，叶子数量为 1
                continue

            # 选择最佳划分特征
            best_feature_name, best_impurity = self.choose_best_feature_to_split(current_X, current_y)
            current_node.feature_name = best_feature_name
            current_node.impurity = best_impurity[0]
            current_node.feature_index = self.columns.index(best_feature_name)
            feature_values = current_X[best_feature_name]

            if len(best_impurity) == 1:  # 离散值特征
                current_node.is_continuous = False
                unique_vals = feature_values.unique()
                sub_X = current_X.drop(best_feature_name, axis=1)

                for value in unique_vals:
                    child_node = Node()
                    child_node.high = current_node.high + 1
                    queue.append((child_node, sub_X[feature_values == value], current_y[feature_values == value]))
                    current_node.subtree[value] = child_node

            elif len(best_impurity) == 2:  # 连续值特征
                current_node.is_continuous = True
                current_node.split_value = best_impurity[1]
                up_part = '>= {:.3f}'.format(current_node.split_value)
                down_part = '< {:.3f}'.format(current_node.split_value)

                child_node_up = Node()
                child_node_down = Node()
                child_node_up.high = current_node.high + 1
                child_node_down.high = current_node.high + 1
                queue.append((child_node_up, current_X[feature_values >= current_node.split_value],
                              current_y[feature_values >= current_node.split_value]))
                queue.append((child_node_down, current_X[feature_values < current_node.split_value],
                              current_y[feature_values < current_node.split_value]))

                current_node.subtree[up_part] = child_node_up
                current_node.subtree[down_part] = child_node_down

        # 逆序遍历 nodes_to_process，计算每个节点的 leaf_num
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.is_leaf:
                node.leaf_num = 1
            else:
                node.leaf_num = sum(child.leaf_num for child in node.subtree.values())

        return root

    def predict(self, X):
        '''
        同样只支持 pd.DataFrame类型数据
        :param X:  pd.DataFrame 类型
        :return:   若
        '''
        if not hasattr(self, "tree_"):
            raise Exception('you have to fit first before predict.')
        if X.ndim == 1:
            return self.predict_single(X)
        else:
            return X.apply(self.predict_single, axis=1)

    def predict_single(self, x, subtree=None):
        '''
        预测单一样本。 实际上这里也可以写成循环，写成递归样本大的时候有栈溢出的风险。
        :param x:
        :param subtree: 根据特征，往下递进的子树。
        :return:
        '''
        if subtree is None:
            subtree = self.tree_

        if subtree.is_leaf:
            return subtree.leaf_class

        if subtree.is_continuous:  # 若是连续值，需要判断是
            if x[subtree.feature_index] >= subtree.split_value:
                return self.predict_single(x, subtree.subtree['>= {:.3f}'.format(subtree.split_value)])
            else:
                return self.predict_single(x, subtree.subtree['< {:.3f}'.format(subtree.split_value)])
        else:
            return self.predict_single(x, subtree.subtree[x[subtree.feature_index]])

    # 选择最佳划分特征
    def choose_best_feature_to_split(self, X, y):
        assert self.criterion in ('gini', 'infogain', 'gainratio')
        if self.criterion == 'gini':
            return self.choose_best_feature_gini(X, y)
        elif self.criterion == 'infogain':
            return self.choose_best_feature_infogain(X, y)
        elif self.criterion == 'gainratio':
            return self.choose_best_feature_gainratio(X, y)

    def choose_best_feature_gini(self, X, y):
        features = X.columns
        best_feature_name = None
        best_gini = [float('inf')]
        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            gini_idex = self.gini_index(X[feature_name], y, is_continuous)
            if gini_idex[0] < best_gini[0]:
                best_feature_name = feature_name
                best_gini = gini_idex

        return best_feature_name, best_gini

    def choose_best_feature_infogain(self, X, y):
        '''
        以返回值中best_info_gain 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益来选择的最佳划分属性，第一个返回值为属性名称，
        '''
        features = X.columns
        best_feature_name = None
        best_info_gain = [float('-inf')]
        entD = self.entroy(y)
        for feature_name in features:  # 分别对于每一种属性
            is_continuous = type_of_target(
                X[feature_name]) == 'continuous'  # 是 scikit-learn 库中的一个函数，用于确定目标变量（标签）的类型。
            info_gain = self.info_gain(X[feature_name], y, entD, is_continuous)
            if info_gain[0] > best_info_gain[0]:
                best_feature_name = feature_name
                best_info_gain = info_gain

        return best_feature_name, best_info_gain

    def choose_best_feature_gainratio(self, X, y):
        '''
        以返回值中best_gain_ratio 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益率来选择的最佳划分属性，第一个返回值为属性名称，第二个为最佳划分属性对应的信息增益率
        '''
        features = X.columns
        best_feature_name = None
        best_gain_ratio = [float('-inf')]
        entD = self.entroy(y)  # 信息熵

        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            info_gain_ratio = self.info_gainRatio(X[feature_name], y, entD, is_continuous)
            if info_gain_ratio[0] > best_gain_ratio[0]:
                best_feature_name = feature_name
                best_gain_ratio = info_gain_ratio

        return best_feature_name, best_gain_ratio

    def gini_index(self, feature, y, is_continuous=False):
        '''
        计算基尼指数， 对于连续值，选择基尼系统最小的点，作为分割点
        -------
        :param feature:
        :param y:
        :return:
        '''
        m = y.shape[0]
        unique_value = pd.unique(feature)
        if is_continuous:
            unique_value.sort()  # 排序, 用于建立分割点
            # 这里其实也可以直接用feature值作为分割点，但这样会出现空集， 所以还是按照书中4.7式建立分割点。好处是不会出现空集
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]

            min_gini = float('inf')
            min_gini_point = None
            for split_point_ in split_point_set:  # 遍历所有的分割点，寻找基尼指数最小的分割点
                Dv1 = y[feature <= split_point_]
                Dv2 = y[feature > split_point_]
                gini_index = Dv1.shape[0] / m * self.gini(Dv1) + Dv2.shape[0] / m * self.gini(Dv2)

                if gini_index < min_gini:
                    min_gini = gini_index
                    min_gini_point = split_point_
            return [min_gini, min_gini_point]
        else:
            gini_index = 0
            for value in unique_value:
                Dv = y[feature == value]
                m_dv = Dv.shape[0]
                gini = self.gini(Dv)  # 原书4.5式
                gini_index += m_dv / m * gini  # 4.6式

            return [gini_index]

    def gini(self, y):
        p = pd.value_counts(y) / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return gini

    def info_gain(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益
        ------
        :param feature: 当前特征（属性）下所有样本值
        :param y:       对应标签值
        :return:        当前特征的信息增益, list类型，若当前特征为离散值则只有一个元素为信息增益，若为连续值，则第一个元素为信息增益，第二个元素为切分点
        '''
        m = y.shape[0]  # y的shape[0]是行数吧，也就是样本数
        unique_value = pd.unique(feature)  # 属性a的av，例如：颜色有乌黑、青绿等
        if is_continuous:  # 如果是连续值的话，需要进行二分（离散化），要根据最小信息熵找出最佳划分点
            unique_value.sort()  # 排序, 用于建立分割点
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in
                               range(len(unique_value) - 1)]  # 在每两个值的区间取中点
            min_ent = float('inf')  # 挑选信息熵最小的分割点
            min_ent_point = None
            for split_point_ in split_point_set:

                Dv1 = y[feature <= split_point_]  # 这是第一类的标签值
                Dv2 = y[feature > split_point_]  # 这是第二类的标签值
                feature_ent_ = Dv1.shape[0] / m * self.entroy(Dv1) + Dv2.shape[0] / m * self.entroy(
                    Dv2)  # 套公式，信息增益公式负号后面的部分，越小越好

                if feature_ent_ < min_ent:
                    min_ent = feature_ent_
                    min_ent_point = split_point_
            gain = entD - min_ent  # 信息增益公式

            return [gain, min_ent_point]  # 返回该特征（属性）的信息增益值以及连续值划分点（二分）

        else:  # 该特征为离散值
            feature_ent = 0  # 信息增益公式负号后面的部分，越小越好
            for value in unique_value:  # 遍历av
                Dv = y[feature == value]  # 当前特征中取值为 value 的样本，即书中的 D^{v}
                feature_ent += Dv.shape[0] / m * self.entroy(Dv)

            gain = entD - feature_ent  # 信息增益公式
            return [gain]

    def info_gainRatio(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益率 参数和info_gain方法中参数一致
        ------
        :param feature:
        :param y:
        :param entD:
        :return:
        '''

        if is_continuous:
            # 对于连续值，以最大化信息增益选择划分点之后，计算信息增益率，注意，在选择划分点之后，需要对信息增益进行修正，要减去log_2(N-1)/|D|，N是当前特征的取值个数，D是总数据量。
            # 修正原因是因为：当离散属性和连续属性并存时，C4.5算法倾向于选择连续特征做最佳树分裂点
            # 信息增益修正中，N的值，网上有些资料认为是“可能分裂点的个数”，也有的是“当前特征的取值个数”，这里采用“当前特征的取值个数”。
            # 这样 (N-1)的值，就是去重后的“分裂点的个数” , 即在info_gain函数中，split_point_set的长度，个人感觉这样更加合理。有时间再看看原论文吧。

            gain, split_point = self.info_gain(feature, y, entD, is_continuous)
            p1 = np.sum(feature <= split_point) / feature.shape[0]  # 小于或划分点的样本占比
            p2 = 1 - p1  # 大于划分点样本占比
            IV = -(p1 * np.log2(p1) + p2 * np.log2(p2))

            grain_ratio = (gain - np.log2(feature.nunique()) / len(y)) / IV  # 对信息增益修正
            return [grain_ratio, split_point]
        else:
            p = pd.value_counts(feature) / feature.shape[0]  # 当前特征下 各取值样本所占比率
            IV = np.sum(-p * np.log2(p))  # 原书4.4式
            grain_ratio = self.info_gain(feature, y, entD, is_continuous)[0] / IV
            return [grain_ratio]

    def entroy(self, y):  # 计算信息熵
        p = pd.value_counts(y) / y.shape[0]  # 计算各类样本所占比率
        ent = np.sum(-p * np.log2(p))
        return ent