# -*- coding: utf-8 -*-
"""
高级可视化与基于 LSTM 的用车需求预测（PyTorch 版本）
---------------------------------------------------
本脚本展示了如何使用伦敦共享单车数据集，通过 **双向 LSTM** 网络预测未来的用车需求。
与最初的 TensorFlow/Keras Notebook 相比，本实现全部改用 **PyTorch**，
并将代码中的注释与文档说明翻译为了中文。

主要步骤：
1. 数据加载
2. 特征工程
3. 探索性数据分析（可视化）
4. 数据预处理
5. 构建与训练双向 LSTM 模型（PyTorch）
6. 预测与可视化
7. 评价指标计算
"""

import warnings
warnings.filterwarnings("ignore")

# ================ 1. 数据加载 ================
import kagglehub                      # 用于在本地 Colab 环境中下载 Kaggle 数据集
import pandas as pd
from pathlib import Path

# 下载数据集并读取
print("正在下载 Kaggle 数据集……")
root_path: Path = kagglehub.dataset_download('hmavrodiev/london-bike-sharing-dataset')
file_path = root_path / 'london_merged.csv'

data = pd.read_csv(file_path)
print("数据加载完毕，前 5 行预览：")
print(data.head())

# ================ 2. 特征工程 ================
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context="notebook", style="darkgrid", palette="deep",
        font="sans-serif", font_scale=1, color_codes=True)

# 将时间戳列转换为日期时间对象并设置为索引
print("开始进行时间特征工程……")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.set_index("timestamp")

# 新增一些常见的时间特征
data["hour"] = data.index.hour
data["day_of_month"] = data.index.day
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
print("特征工程完成，当前列：", data.columns.tolist())

# ================ 3. 探索性数据分析 (EDA) ================
print("绘制相关性热力图……")
plt.figure(figsize=(16, 6))
sns.heatmap(data.corr(), cmap="YlGnBu", square=True, linewidths=.5,
            center=0, linecolor="red")
plt.title("特征相关性热力图")
plt.show()

# 其他可视化示例（可按需解注释）
# plt.figure(figsize=(16,6))
# sns.lineplot(data=data, x=data.index, y=data.cnt)
# plt.title("按时间序列的用车需求")
# plt.xticks(rotation=90)
# plt.show()

# ================ 4. 数据预处理 ================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

tqdm().pandas()

print("开始划分训练集与测试集……")
train, test = train_test_split(data, test_size=0.1, random_state=0)

# 对数值型特征进行 MinMax 缩放
num_cols = ['t1', 't2', 'hum', 'wind_speed']
scaler_features = MinMaxScaler()
train.loc[:, num_cols] = scaler_features.fit_transform(train[num_cols])
test.loc[:, num_cols] = scaler_features.transform(test[num_cols])

# 对目标变量 cnt 进行独立缩放
scaler_cnt = MinMaxScaler()
train['cnt'] = scaler_cnt.fit_transform(train[['cnt']])
test['cnt'] = scaler_cnt.transform(test[['cnt']])

# 生成监督学习所需的时序数据 (X, y)

def prepare_data(X: pd.DataFrame, y: pd.Series, time_steps: int = 24):
    """将多变量时序数据切片，生成模型输入输出。"""
    Xs, Ys = [], []
    for i in tqdm(range(len(X) - time_steps)):
        v = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(v)
        Ys.append(y.iloc[i + time_steps])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

TIME_STEPS = 24
print("准备训练/测试样本……")
X_train, y_train = prepare_data(train, train.cnt, TIME_STEPS)
X_test, y_test = prepare_data(test, test.cnt, TIME_STEPS)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

# ================ 5. 构建与训练 PyTorch LSTM 模型 ================
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 检测 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

class BiLSTMModel(nn.Module):
    """双向 LSTM 网络定义。
    输入张量尺寸: (batch, seq_len, features)
    输出: 单个归一化数值 (0 ~ 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)           # 取最后一个时间步输出
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc(out)
        return self.act(out).squeeze()

model = BiLSTMModel(input_dim=X_train.shape[2]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 构建 DataLoader
BATCH_SIZE = 32
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_EPOCHS = 100
train_losses, val_losses = [], []
print("开始模型训练……")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]\t训练损失: {epoch_loss:.4f}\t验证损失: {val_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="训练损失")
plt.plot(val_losses, label="验证损失")
plt.xlabel("轮数 (Epoch)")
plt.ylabel("MSE")
plt.legend()
plt.title("训练与验证损失曲线")
plt.show()

# ================ 6. 预测与可视化 ================
print("开始生成测试集预测……")
model.eval()
with torch.no_grad():
    test_pred = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()

y_test_inv = scaler_cnt.inverse_transform(y_test.reshape(-1, 1))
pred_inv = scaler_cnt.inverse_transform(test_pred.reshape(-1, 1))

plt.figure(figsize=(16, 6))
plt.plot(y_test_inv.flatten(), marker=".", label="真实值")
plt.plot(pred_inv.flatten(), marker=".", label="预测值", color="r")
plt.title("用车需求预测结果 (逆缩放后)")
plt.legend()
plt.show()

# ================ 7. 模型评价 ================
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
r2 = r2_score(y_test_inv, pred_inv)
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

# 将真实与预测结果保存为 DataFrame 便于后续分析
result_df = pd.DataFrame({
    "actual": y_test_inv.flatten(),
    "predicted": pred_inv.flatten()
})
print(result_df.head())

plt.figure(figsize=(16, 6))
plt.plot(result_df.actual, label="真实值")
plt.plot(result_df.predicted, label="预测值")
plt.title("真实值 vs. 预测值 对比")
plt.legend()
plt.show()

print("脚本运行完毕，感谢使用！")