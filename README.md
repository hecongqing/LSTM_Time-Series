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

## 两种建模场景

### 1. 预测下一个小时（单步预测）

- 训练指令：
  ```bash
  python src/train.py \
    --data-path data/raw/london_merged.csv \
    --pred-horizon 1 \
    --epochs 30 \
    --save-dir outputs/1h
  ```
- 评估指标将仅对 `t+1` 的租借量进行计算。
- 部署时在 Streamlit 侧边栏选择 `Prediction horizon = 1` 并将模型路径改为 `outputs/1h/model_best.pt`。

### 2. 预测未来 6 小时（多步预测）

- 训练指令：
  ```bash
  python src/train.py \
    --data-path data/raw/london_merged.csv \
    --pred-horizon 6 \
    --epochs 30 \
    --save-dir outputs/6h
  ```
- 模型输出为一个长度 6 的向量，分别对应 `t+1 ... t+6`。
- 部署时在 Streamlit 选择 `Prediction horizon = 6` 并切换模型路径。

---

## 部署与可视化

安装额外依赖：
```bash
pip install streamlit
```
运行应用：
```bash
streamlit run src/app.py --server.port 8501
```
然后在浏览器打开 `http://localhost:8501` 即可：

1. 侧边栏配置模型权重、Scaler、数据集路径以及预测步长。
2. 拖动滑块查看不同样本的 **真实 vs 预测** 对比曲线与 RMSE/MAE 指标。
3. 页面下方展示 LSTM 网络结构文本摘要。

![](docs/screenshot.png)

> 生产环境可将 Streamlit 部署到云服务器或封装为 Docker 镜像。

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

## ONNX 部署（FastAPI + ONNX Runtime）

为了在生产环境中获得更高的推理性能并与语言无关，本项目提供了 **ONNX 导出与推理 API** 示例，结合 FastAPI & onnxruntime 轻量级上线。

1. **导出模型至 ONNX**（假设已完成模型训练并得到 `model_best.pt`）：

   ```bash
   python deploy/convert_to_onnx.py \
     --model-path outputs/model_best.pt \
     --onnx-path outputs/model_best.onnx \
     --input-dim 13 --window-size 24 --pred-horizon 6
   ```

   运行完成后将在 `outputs/` 生成 `model_best.onnx` 文件。

2. **安装推理依赖**（若已按 `requirements.txt` 全量安装，可跳过）：

   ```bash
   pip install onnxruntime fastapi uvicorn[standard]
   ```

3. **启动 FastAPI 推理服务**（默认为 CPU，可根据环境修改 `providers` 列表以启用 CUDA / TensorRT 等）：

   ```bash
   python -m uvicorn deploy.onnx_server:app --host 0.0.0.0 --port 8000
   ```

   服务启动后，可通过 `POST /predict` 发送形状为 `(WINDOW_SIZE, INPUT_DIM)` 的二维列表（JSON）获得预测结果：

   ```json
   {
     "data": [[0.42, 0.13, ...], [...], ...]
   }
   ```

   返回：

   ```json
   { "prediction": [0.21, 0.19, ...] }
   ```

4. **在 Streamlit 前端接入远程推理**：

   打开 `src/app.py` 对应的 Streamlit 应用，在侧边栏的 **Inference API URL** 输入运行中的 FastAPI 地址（如 `http://localhost:8000`），即可实时调用 ONNX 服务并展示预测结果。

> 如需部署到容器或云环境，可将 `deploy/onnx_server.py` 打包为 Docker 镜像，或挂载 NGINX / Traefik 进行反向代理与负载均衡。

---

## 参考文献

* [London Bike Sharing Dataset on Kaggle](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)
* 深度学习与时间序列预测相关书籍/论文

---

如有问题或改进建议，欢迎提 issue 或 PR 🙌