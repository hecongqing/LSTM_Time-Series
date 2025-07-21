# London Bike Sharing Demand Prediction with LSTM (PyTorch)

本项目基于 Kaggle 开源的 **London Bike Sharing Dataset**，使用 PyTorch 实现 LSTM 循环神经网络来预测未来若干小时共享单车的租借需求。

## 目录结构

```text
.
├── data/                   # 原始及处理后的数据
├── notebooks/              # 交互式探索分析
├── outputs/                # 训练日志、模型权重、结果图表
├── src/                    # 主要源代码
│   ├── dataset.py          # 数据加载与滑窗 Dataset
│   ├── model.py            # LSTM 网络结构
│   ├── train.py            # 训练入口脚本
│   └── utils.py            # 工具函数
├── requirements.txt        # 依赖列表
└── README.md               # 当前文档
```

## 快速开始

1. 克隆仓库并安装依赖：

```bash
pip install -r requirements.txt
```

2. 下载数据集（约 1.3MB）：

```bash
mkdir -p data/raw && 
wget -O data/raw/london_merged.csv \
  https://raw.githubusercontent.com/hmavrodiev/london-bike-sharing-dataset/master/london_merged.csv
```

3. 训练模型：

```bash
python src/train.py \
  --data-path data/raw/london_merged.csv \
  --epochs 30 \
  --pred-horizon 6 \
  --save-dir outputs
```

默认使用 CPU；如需 GPU，请确保正确安装了 CUDA 版本的 PyTorch，并在运行时传入 `--device cuda`。

4. 训练完成后，可在 `outputs/` 查看：

* `model_best.pt` — 验证集表现最佳的模型权重
* `loss_curve.png` — 训练/验证损失曲线
* `prediction_plot.png` — 在测试集上的真实 vs. 预测曲线

## 部署示例

经过训练的模型可轻松集成进后端或 Web 应用。示例：

```python
from src.model import LSTMBikePredictor
import torch, joblib, pandas as pd

model = LSTMBikePredictor(input_dim=10, hidden_dim=64, num_layers=2, output_dim=6)
model.load_state_dict(torch.load("outputs/model_best.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("outputs/feature_scaler.pkl")

# 假设 `latest_df` 为最近 24 小时已处理好特征的 DataFrame
x = torch.tensor(scaler.transform(latest_df.values), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    preds = model(x)
print(preds)
```

## 参考文献

* [London Bike Sharing Dataset on Kaggle](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)
* 深度学习与时间序列预测相关书籍/论文

---

如有问题或改进建议，欢迎提 issue 或 PR 🙌