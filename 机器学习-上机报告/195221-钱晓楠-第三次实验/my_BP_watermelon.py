import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 固定随机种子，以确保结果的可重复性
np.random.seed(42)

# 读取西瓜数据集3.0
data = pd.read_csv(r"E:\Python_file\MachineLearning2\watermelon3_0_Ch.csv")

# 将类别标签转换为数值。将'好瓜'列中的“是”映射为1，“否”映射为0
data['好瓜'] = data['好瓜'].map({'是': 1, '否': 0})

# 提取特征和标签
X = data.iloc[:, 1:-1]  # 提取特征列，忽略第一列（编号）和最后一列（标签）
y = data.iloc[:, -1].values  # 提取标签列并转为数组

# 对非数值特征进行整数编码，方便神经网络处理
for column in X.select_dtypes(include=['object']).columns:
    X[column] = pd.factorize(X[column])[0] + 1  # 整数编码从1开始，避免0值
X = X.values  # 将特征数据转换为NumPy数组，方便后续计算
print(X)  # 输出编码后的特征数据

y = y.reshape(-1, 1)  # 将标签转换为二维数组，便于后续计算

# 定义 BP 神经网络类
class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化 BP 神经网络
        参数：
        - input_size：输入层节点数
        - hidden_size：隐藏层节点数
        - output_size：输出层节点数
        - learning_rate：学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 随机初始化权重矩阵
        self.v = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.gamma = np.zeros((1, hidden_size))            # 隐藏层的偏置
        self.w = np.random.randn(hidden_size, output_size) # 隐藏层到输出层的权重
        self.theta = np.zeros((1, output_size))            # 输出层的偏置

    def sigmoid(self, x):
        """
        Sigmoid 激活函数，将输入值映射到 (0, 1) 范围
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        前向传播：计算从输入层到隐藏层，再到输出层的输出值
        参数：
        - X：输入数据
        返回：
        - y_hat：网络的输出（预测值）
        """
        self.alpha = np.dot(X, self.v) + self.gamma       # 计算隐藏层加权输入
        self.b = self.sigmoid(self.alpha)                 # 计算隐藏层输出
        self.beta = np.dot(self.b, self.w) + self.theta   # 计算输出层加权输入
        self.y_hat = self.sigmoid(self.beta)              # 计算输出层输出（预测值）
        return self.y_hat

    def backward(self, X, y, y_hat):
        # 计算输出层误差项
        self.g = y_hat * (1 - y_hat) * (y - y_hat)
        # 计算隐藏层误差项
        self.e = self.b * (1 - self.b) * np.dot(self.g, self.w.T)

        # 更新权重和偏置
        self.w += self.learning_rate * np.dot(self.b.T, self.g)  # 更新隐藏层到输出层的权重
        self.theta += self.learning_rate * np.sum(self.g, axis=0, keepdims=True)  # 更新输出层偏置
        self.v += self.learning_rate * np.dot(X.T, self.e)  # 更新输入层到隐藏层的权重
        self.gamma += self.learning_rate * np.sum(self.e, axis=0, keepdims=True)  # 更新隐藏层偏置

    def train(self, X, y, epochs=1000):
        """
        训练模型，通过多次迭代进行前向和反向传播
        参数：
        - X：训练数据
        - y：训练标签
        - epochs：迭代次数
        """
        for i in range(epochs):
            y_hat = self.forward(X)  # 前向传播
            self.backward(X, y, y_hat)  # 反向传播

    def predict(self, X):
        """
        预测函数：返回给定输入的预测结果
        参数：
        - X：输入数据
        返回：
        - 前向传播的预测值
        """
        return self.forward(X)


# 初始化 BP 神经网络参数
input_size = X.shape[1]  # 输入特征数
hidden_size = 3  # 隐藏层节点数
output_size = 1  # 输出层节点数（二分类）
learning_rate = 0.1  # 学习率
epochs = 1000  # 训练迭代次数

# 测试集比例列表，用于不同的数据集划分
test_sizes = [0.1, 0.2, 0.25, 0.3]

for test_size in test_sizes:
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 初始化并训练 BP 神经网络
    bpnn = BPNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    bpnn.train(X_train, y_train, epochs)  # 在训练集上训练模型

    # 在测试集上进行预测并转换为二值（0 或 1）
    predictions = (bpnn.predict(X_test) > 0.5).astype(int)

    # 计算并输出评估指标
    accuracy = accuracy_score(y_test, predictions)  # 准确率
    precision = precision_score(y_test, predictions)  # 精确率
    recall = recall_score(y_test, predictions)  # 召回率
    f1 = f1_score(y_test, predictions)  # F1分数

    # 输出评估结果
    print(f'Test Size: {test_size}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%\n')
