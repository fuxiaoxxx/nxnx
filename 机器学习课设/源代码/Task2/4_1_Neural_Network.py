import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('creditcard.csv')

# 特征和标签
X = data.drop('Class', axis=1)
y = data['Class']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用SMOTE过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 使用随机欠采样（Random UnderSampling）
undersample = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersample.fit_resample(X_train_scaled, y_train)


# 将数据转换为PyTorch张量
def to_tensor(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    return X_tensor, y_tensor


# 神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# 训练函数
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 评估模型
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).float()  # 将输出转换为0或1
    return y_pred


# 训练和评估神经网络模型（使用SMOTE过采样后的数据）
X_train_smote_tensor, y_train_smote_tensor = to_tensor(X_train_smote, y_train_smote)
X_test_scaled_tensor, y_test_tensor = to_tensor(X_test_scaled, y_test)

model_smote = SimpleNN(X_train_smote_tensor.shape[1])
train_model(model_smote, X_train_smote_tensor, y_train_smote_tensor, epochs=10, batch_size=32)

y_pred_smote = evaluate_model(model_smote, X_test_scaled_tensor, y_test_tensor)
print("SMOTE over-sampling classification report:")
print(classification_report(y_test, y_pred_smote.numpy()))

print("SMOTE over-sampling confusion matrix:")
print(confusion_matrix(y_test, y_pred_smote.numpy()))

# 绘制训练过程中的损失变化（损失在PyTorch中通过print显示，所以不需要保存到history）
# 可以添加可视化内容或者保存数据

# 训练和评估神经网络模型（使用随机欠采样后的数据）
X_train_under_tensor, y_train_under_tensor = to_tensor(X_train_under, y_train_under)

model_under = SimpleNN(X_train_under_tensor.shape[1])
train_model(model_under, X_train_under_tensor, y_train_under_tensor, epochs=10, batch_size=32)

y_pred_under = evaluate_model(model_under, X_test_scaled_tensor, y_test_tensor)
print("Random Under-sampling classification report:")
print(classification_report(y_test, y_pred_under.numpy()))

print("Random Under-sampling confusion matrix:")
print(confusion_matrix(y_test, y_pred_under.numpy()))
