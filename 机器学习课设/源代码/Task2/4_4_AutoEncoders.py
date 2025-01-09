import torch
import torch.nn as nn
import torch.optim as optim
from pyomo.contrib.parmest.graphics import sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置随机种子，保证结果可复现
np.random.seed(203)
torch.manual_seed(203)

# 加载数据集
data = pd.read_csv("creditcard.csv")
data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)  # 将时间列转换为小时制

# 分析类别分布
vc = data['Class'].value_counts().to_frame().reset_index()
vc['percent'] = vc["Class"].apply(lambda x: round(100 * float(x) / len(data), 2))
vc = vc.rename(columns={"index": "Target", "Class": "Count"})

# 对数据进行采样
non_fraud = data[data['Class'] == 0].sample(1000)  # 随机采样1000条非欺诈数据
fraud = data[data['Class'] == 1]  # 获取所有欺诈数据

df = non_fraud._append(fraud).sample(frac=1).reset_index(drop=True)  # 合并并打乱数据
X = df.drop(['Class'], axis=1).values  # 特征
Y = df["Class"].values  # 标签

# t-SNE可视化函数
def tsne_plot(x1, y1, title="t-SNE Plot"):
    tsne = TSNE(n_components=2, random_state=0)  # 初始化t-SNE降维器
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', label='Non Fraud')  # 非欺诈点
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', label='Fraud')  # 欺诈点

    plt.legend(loc='best')
    plt.title(title)
    plt.show()

# 原始数据的t-SNE可视化
tsne_plot(X, Y, "Original Data")

# PyTorch实现的Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # 编码部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.ReLU()
        )
        # 解码部分
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)  # 编码
        decoded = self.decoder(encoded)  # 解码
        return decoded

# 数据预处理
x = data.drop(["Class"], axis=1).values  # 提取特征
y = data["Class"].values  # 提取标签
x_scale = preprocessing.MinMaxScaler().fit_transform(x)  # 特征归一化
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]  # 按标签分开

# 训练Autoencoder
input_dim = x_scale.shape[1]  # 输入特征维度
model = Autoencoder(input_dim)  # 初始化模型
criterion = nn.MSELoss()  # 损失函数为均方误差
optimizer = optim.Adadelta(model.parameters(), lr=1.0)  # 使用Adadelta优化器

x_train = torch.tensor(x_norm[:3000], dtype=torch.float32)  # 选取部分非欺诈数据作为训练集
x_val = torch.tensor(x_fraud, dtype=torch.float32)  # 全部欺诈数据作为验证集

num_epochs = 50  # 训练轮次
batch_size = 64  # 批次大小

# 定义 L1 正则化函数
def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# 更新训练循环，加入正则化损失
lambda_l1 = 10e-5  # 与 Keras 中的值保持一致
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 清空梯度
    outputs = model(x_train)  # 前向传播
    mse_loss = criterion(outputs, x_train)  # 计算 MSE 损失
    reg_loss = l1_regularization(model, lambda_l1)  # 计算 L1 正则化损失
    loss = mse_loss + reg_loss  # 总损失 = MSE + L1
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {reg_loss.item():.4f}")

# 提取隐藏层表示
model.eval()  # 设置为评估模式
with torch.no_grad():
    norm_hid_rep = model.encoder(x_train).numpy()  # 非欺诈数据的隐藏层表示
    fraud_hid_rep = model.encoder(x_val).numpy()  # 欺诈数据的隐藏层表示

rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis=0)  # 合并隐藏层表示
y_n = np.zeros(norm_hid_rep.shape[0])  # 非欺诈标签
y_f = np.ones(fraud_hid_rep.shape[0])  # 欺诈标签
rep_y = np.append(y_n, y_f)

# 隐藏层表示的t-SNE可视化
tsne_plot(rep_x, rep_y, "Latent Representations")

# 使用隐藏层表示进行逻辑回归分类
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)  # 划分训练集和验证集
clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)  # 训练逻辑回归模型
pred_y = clf.predict(val_x)  # 预测

# 分类结果评估
print("Classification Report:")
print(classification_report(val_y, pred_y))  # 打印分类报告
print("Accuracy Score:", accuracy_score(val_y, pred_y))  # 打印准确率

# 使用隐藏层表示进行逻辑回归分类
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)  # 划分训练集和验证集
clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)  # 训练逻辑回归模型
pred_y = clf.predict(val_x)  # 预测

# 分类结果评估
print("Classification Report:")
print(classification_report(val_y, pred_y))  # 打印分类报告
print("Accuracy Score:", accuracy_score(val_y, pred_y))  # 打印准确率


