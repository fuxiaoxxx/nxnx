import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 固定随机种子
np.random.seed(42)

# 加载 UCI 乳腺癌数据集
data = load_breast_cancer()
X = data.data  # 特征
y = data.target.reshape(-1, 1)  # 标签，并转换成列向量

# 定义 BP 神经网络类
class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.v = np.random.randn(input_size, hidden_size)
        self.gamma = np.zeros((1, hidden_size))
        self.w = np.random.randn(hidden_size, output_size)
        self.theta = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.alpha = np.dot(X, self.v) + self.gamma
        self.b = self.sigmoid(self.alpha)
        self.beta = np.dot(self.b, self.w) + self.theta
        self.y_hat = self.sigmoid(self.beta)
        return self.y_hat

    def backward(self, X, y, y_hat):
        self.g = y_hat * (1 - y_hat) * (y - y_hat)
        self.e = self.b * (1 - self.b) * np.dot(self.g, self.w.T)

        self.w += self.learning_rate * np.dot(self.b.T, self.g)
        self.theta += self.learning_rate * np.sum(self.g, axis=0, keepdims=True)
        self.v += self.learning_rate * np.dot(X.T, self.e)
        self.gamma += self.learning_rate * np.sum(self.e, axis=0, keepdims=True)

    def train(self, X, y, max_epochs=1000, error_threshold=0.01):
        for i in range(max_epochs):
            y_hat = self.forward(X)
            error = np.mean(0.5 * (y - y_hat) ** 2)  # 均方误差

            if error < error_threshold:
                print(f"Training stopped at epoch {i} with error: {error:.5f}")
                break

            self.backward(X, y, y_hat)

    def predict(self, X):
        return self.forward(X)

# 初始化 BP 神经网络参数
input_size = X.shape[1]
hidden_size = 10  # 增大隐藏层神经元数量以适应更复杂的数据集
output_size = 1
learning_rate = 0.1
max_epochs = 10000
error_threshold = 0.02

# 测试集比例
test_sizes = [0.1, 0.2, 0.25, 0.3]

for test_size in test_sizes:
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 初始化并训练 BP 神经网络
    bpnn = BPNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    bpnn.train(X_train, y_train, max_epochs, error_threshold)

    # 预测并转换为二值
    predictions = (bpnn.predict(X_test) > 0.5).astype(int)

    # 计算并输出评估指标
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f'Test Size: {test_size}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%\n')
