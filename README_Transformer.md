# Transformer 时间序列预测模型

本项目新增了基于Transformer架构的时间序列预测功能，与现有的LSTM模型保持相同的代码结构。

## 📁 新增文件结构

```
src/
├── transformer_model.py        # Transformer模型定义
├── transformer_dataset.py      # Transformer数据处理
├── transformer_train.py        # Transformer训练脚本
├── transformer_app.py          # Transformer Streamlit应用
└── transformer_example.py      # 使用示例脚本
```

## 🚀 快速开始

### 1. 安装依赖

确保已安装所有必需的依赖包：

```bash
pip install torch torchvision torchaudio
pip install streamlit pandas numpy scikit-learn matplotlib joblib tqdm
```

### 2. 训练Transformer模型

#### 方法一：使用示例脚本（推荐）

```bash
cd src
python transformer_example.py
```

然后选择选项1开始训练。

#### 方法二：使用命令行

```bash
cd src
python transformer_train.py --data-path ../data/raw/london_merged.csv \
                            --epochs 50 \
                            --batch-size 64 \
                            --lr 3e-4 \
                            --embed-size 256 \
                            --num-heads 8 \
                            --num-blocks 2 \
                            --save-dir transformer_outputs
```

### 3. 运行Streamlit应用

```bash
cd src
streamlit run transformer_app.py
```

## 🏗️ 模型架构

### Transformer模型特点

- **自注意力机制**: 能够并行处理序列中的所有位置
- **多头注意力**: 同时关注不同类型的模式
- **位置编码**: 通过时间特征嵌入提供位置信息
- **层归一化**: 稳定训练过程
- **残差连接**: 缓解梯度消失问题

### 两种模型变体

#### 1. 简化版本 (SimplifiedTransformerBikePredictor)
- 仅使用数值特征
- 适合快速实验和资源受限环境
- 参数更少，训练更快

#### 2. 完整版本 (TransformerBikePredictor)
- 支持分类特征和静态特征
- 使用嵌入层处理分类变量
- 更强的表达能力

## 📊 数据处理

### 特征工程

```python
# 数值特征（连续变量）
numeric_features = [
    "cnt",           # 目标变量：自行车数量
    "t1", "t2",      # 温度特征
    "hum",           # 湿度
    "wind_speed",    # 风速
    "hour_sin", "hour_cos",         # 时间周期编码
    "month_sin", "month_cos",       # 月份周期编码
    "dayofweek_sin", "dayofweek_cos"  # 星期周期编码
]

# 分类特征（可选）
categorical_features = [
    "hour",          # 小时（0-23）
    "dayofweek",     # 星期（0-6）
    "month",         # 月份（1-12）
    "weather_code",  # 天气编码
    "season"         # 季节
]

# 静态特征（可选）
static_features = [
    "is_holiday",    # 是否假日
    "is_weekend"     # 是否周末
]
```

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--lr` | 3e-4 | 学习率 |
| `--window-size` | 24 | 输入序列长度（小时） |
| `--pred-horizon` | 6 | 预测长度（小时） |
| `--embed-size` | 256 | 嵌入维度 |
| `--num-heads` | 8 | 注意力头数 |
| `--num-blocks` | 2 | Transformer块数 |
| `--dropout` | 0.1 | Dropout概率 |
| `--use-categorical` | False | 是否使用分类特征 |
| `--loss-type` | mse | 损失函数类型 (mse/rmsle) |

## 📈 性能比较

### Transformer vs LSTM

| 特性 | Transformer | LSTM |
|------|-------------|------|
| **并行处理** | ✅ 支持 | ❌ 序列处理 |
| **长距离依赖** | ✅ 自注意力机制 | ⚠️ 可能梯度消失 |
| **训练速度** | ✅ 更快（并行） | ⚠️ 较慢（序列） |
| **内存使用** | ⚠️ 注意力矩阵占用 | ✅ 较少 |
| **可解释性** | ✅ 注意力权重 | ❌ 黑盒 |
| **参数量** | ⚠️ 相对较多 | ✅ 相对较少 |

## 🔧 使用示例

### 训练模型

```python
from src.transformer_train import train_model

class Args:
    data_path = "data/raw/london_merged.csv"
    epochs = 50
    batch_size = 64
    lr = 3e-4
    embed_size = 256
    num_heads = 8
    num_blocks = 2
    save_dir = "transformer_outputs"
    use_categorical = False

args = Args()
train_model(args)
```

### 加载和使用模型

```python
import torch
import joblib
from src.transformer_model import SimplifiedTransformerBikePredictor

# 加载模型配置
config = joblib.load("transformer_outputs/transformer_model_config.pkl")
scaler = joblib.load("transformer_outputs/transformer_feature_scaler.pkl")

# 创建模型
model = SimplifiedTransformerBikePredictor(
    input_dim=config['num_numeric_features'],
    embed_size=config['embed_size'],
    num_heads=config['num_heads'],
    num_blocks=config['num_blocks'],
    output_dim=config['output_dim']
)

# 加载权重
model.load_state_dict(torch.load("transformer_outputs/transformer_model_best.pt"))
model.eval()

# 预测
with torch.no_grad():
    predictions = model(input_tensor)
```

## 🎯 应用场景

### 适合使用Transformer的情况：
- 需要捕获长期依赖关系
- 有足够的计算资源
- 希望利用注意力机制的可解释性
- 数据量较大，能充分利用Transformer的表达能力

### 适合使用LSTM的情况：
- 计算资源有限
- 序列较短
- 需要在线实时预测（内存效率要求高）
- 数据量相对较小

## 📝 输出文件

训练完成后，会在指定目录生成以下文件：

```
transformer_outputs/
├── transformer_model_best.pt           # 最佳模型权重
├── transformer_feature_scaler.pkl      # 特征缩放器
├── transformer_model_config.pkl        # 模型配置
├── transformer_loss_curve.png          # 训练损失曲线
└── transformer_prediction_plot.png     # 预测结果可视化
```

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减小`batch_size`、`embed_size`或`window_size`

### Q: 训练时间太长？
A: 减少`num_blocks`、`embed_size`或`epochs`

### Q: 模型效果不好？
A: 尝试：
- 增加`embed_size`和`num_heads`
- 使用`--use-categorical`启用分类特征
- 调整学习率和训练轮数
- 使用RMSLE损失函数（`--loss-type rmsle`）

### Q: 如何选择超参数？
A: 推荐配置：
- 小数据集：embed_size=128, num_heads=4, num_blocks=1
- 中等数据集：embed_size=256, num_heads=8, num_blocks=2
- 大数据集：embed_size=512, num_heads=16, num_blocks=4

## 🔄 与LSTM版本对比

| 组件 | LSTM版本 | Transformer版本 |
|------|----------|-----------------|
| 模型文件 | `model.py` | `transformer_model.py` |
| 数据处理 | `dataset.py` | `transformer_dataset.py` |
| 训练脚本 | `train.py` | `transformer_train.py` |
| 应用界面 | `app.py` | `transformer_app.py` |
| 输出目录 | `outputs/` | `transformer_outputs/` |

两个版本可以并存使用，互不干扰。