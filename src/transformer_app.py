import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import matplotlib.pyplot as plt

from transformer_dataset import load_and_process_transformer, TransformerBikeDataset, create_features_transformer
from transformer_model import SimplifiedTransformerBikePredictor, TransformerBikePredictor

st.set_page_config(page_title="London Bike Demand Predictor - Transformer", layout="wide")

st.title("🚲 London Bike Sharing Demand Prediction - Transformer Model")

st.markdown(
    """使用训练好的 PyTorch Transformer 模型，预测伦敦共享单车未来需求量，并可视化模型结构与预测效果。"""
)

# Sidebar configuration
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input(
    "Model weights (.pt)", 
    value="transformer_outputs/transformer_model_best.pt", 
    help="Transformer模型权重路径"
)
scaler_path = st.sidebar.text_input(
    "Scaler pickle (.pkl)", 
    value="transformer_outputs/transformer_feature_scaler.pkl"
)
config_path = st.sidebar.text_input(
    "Model config (.pkl)", 
    value="transformer_outputs/transformer_model_config.pkl"
)
data_path = st.sidebar.text_input(
    "Dataset CSV", 
    value="data/raw/london_merged.csv"
)
pred_horizon = st.sidebar.selectbox("Prediction horizon", options=[1, 6], index=1)
window_size = st.sidebar.number_input("Window size (hours)", value=24, step=1)
device = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)

# Load model configuration
@st.cache_resource
def load_model_config(path: str):
    if Path(path).exists():
        return joblib.load(path)
    return None

# Load scaler
@st.cache_resource
def load_scaler(path: str):
    if Path(path).exists():
        return joblib.load(path)
    return None

# Load model
@st.cache_resource
def load_model(model_path: str, config: dict, device: str):
    if not Path(model_path).exists() or config is None:
        return None
        
    if config['model_type'] == 'simplified':
        model = SimplifiedTransformerBikePredictor(
            input_dim=config['num_numeric_features'],
            embed_size=config['embed_size'],
            num_heads=config['num_heads'],
            num_blocks=config['num_blocks'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
        )
    else:
        model = TransformerBikePredictor(
            numeric_features=config['num_numeric_features'],
            categorical_features_dims=config['categorical_dims'],
            static_features_dims=config['static_dims'],
            embed_size=config['embed_size'],
            num_heads=config['num_heads'],
            num_blocks=config['num_blocks'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
        )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Data loading
@st.cache_data
def prepare_data(csv_path: str):
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    df = create_features_transformer(df)
    return df

# Load all components
config = load_model_config(config_path)
scaler = load_scaler(scaler_path)
df_full = prepare_data(data_path)

if df_full is None:
    st.error("Dataset path invalid")
    st.stop()

if config is None:
    st.error("Model configuration not found. Please train the model first.")
    st.stop()
    
if scaler is None:
    st.error("Scaler file not found. Please train the model first.")
    st.stop()

model = load_model(model_path, config, device)
if model is None:
    st.error("Model loading failed. Please check the model path and configuration.")
    st.stop()

# Prepare features
numeric_cols = [
    "cnt",  # target at index 0
    "t1",
    "t2", 
    "hum",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "month_sin", 
    "month_cos",
    "dayofweek_sin",
    "dayofweek_cos",
]

categorical_cols = [
    "hour",
    "dayofweek", 
    "month",
    "weather_code",
    "season",
] if config['model_type'] != 'simplified' else []

static_cols = [
    "is_holiday",
    "is_weekend",
] if config['model_type'] != 'simplified' else []

# Process data
numeric_data = df_full[numeric_cols].values
numeric_scaled = scaler.transform(numeric_data)

categorical_data = None
static_data = None

if config['model_type'] != 'simplified':
    from sklearn.preprocessing import LabelEncoder
    
    if categorical_cols:
        categorical_data = np.zeros((len(df_full), len(categorical_cols)))
        for i, col in enumerate(categorical_cols):
            le = LabelEncoder()
            categorical_data[:, i] = le.fit_transform(df_full[col])
            
    if static_cols:
        static_data = np.zeros((len(df_full), len(static_cols)))
        for i, col in enumerate(static_cols):
            le = LabelEncoder()
            static_data[:, i] = le.fit_transform(df_full[col])

# Create dataset
ds = TransformerBikeDataset(
    numeric_scaled, 
    categorical_data, 
    static_data, 
    window_size, 
    pred_horizon
)

# Display model info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"Model type: {config['model_type']}")
st.sidebar.write(f"Embedding size: {config['embed_size']}")
st.sidebar.write(f"Number of heads: {config['num_heads']}")
st.sidebar.write(f"Number of blocks: {config['num_blocks']}")
st.sidebar.write(f"Dropout: {config['dropout']}")

# Time selection
window_end_timestamps = (
    df_full["timestamp"].iloc[window_size - 1 : len(df_full) - pred_horizon].dt.strftime("%Y-%m-%d %H:%M")
)
timestamp_options = window_end_timestamps.tolist()

selected_time = st.sidebar.selectbox(
    "选择时间窗口（窗口结束时间）", 
    options=timestamp_options, 
    index=0
)
idx = timestamp_options.index(selected_time)

# Make prediction
x_numeric, x_categorical, x_static, y_true = ds[idx]

with torch.no_grad():
    x_numeric_batch = x_numeric.unsqueeze(0).to(device)
    
    if config['model_type'] == 'simplified':
        y_pred = model(x_numeric_batch).cpu().numpy().flatten()
    else:
        x_categorical_batch = None
        x_static_batch = None
        
        if x_categorical is not None:
            x_categorical_batch = x_categorical.unsqueeze(0).to(device)
        if x_static is not None:
            x_static_batch = x_static.unsqueeze(0).to(device)
            
        y_pred = model(x_numeric_batch, x_categorical_batch, x_static_batch).cpu().numpy().flatten()

y_true = y_true.numpy()[:pred_horizon]

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction vs. Ground Truth")
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = range(pred_horizon)
    ax.plot(hours, y_true, 'o-', label="True", linewidth=2, markersize=6)
    ax.plot(hours, y_pred, 's-', label="Predicted", linewidth=2, markersize=6)
    ax.set_xlabel("Future Hour")
    ax.set_ylabel("Scaled Bike Count")
    ax.set_title(f"Transformer Model - {pred_horizon}h Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("Metrics")
    rmse_val = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae_val = np.mean(np.abs(y_pred - y_true))
    mape_val = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    st.metric("RMSE", f"{rmse_val:.4f}")
    st.metric("MAE", f"{mae_val:.4f}")
    st.metric("MAPE", f"{mape_val:.2f}%")
    
    # Feature importance (attention weights could be shown here in future)
    st.subheader("Model Architecture")
    st.write("**Transformer Attention-based Model**")
    st.write(f"- Input sequence length: {window_size}")
    st.write(f"- Prediction horizon: {pred_horizon}")
    st.write(f"- Embedding dimension: {config['embed_size']}")
    st.write(f"- Attention heads: {config['num_heads']}")
    st.write(f"- Transformer blocks: {config['num_blocks']}")

st.markdown("---")

# Additional visualizations
st.subheader("Input Sequence Analysis")

col3, col4 = st.columns(2)

with col3:
    st.write("**Input Numeric Features (Last 24h)**")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main target variable
    target_seq = x_numeric[:, 0].numpy()  # cnt column
    ax.plot(range(window_size), target_seq, 'b-', linewidth=2, label='Historical Count')
    ax.axvline(x=window_size-1, color='r', linestyle='--', alpha=0.7, label='Prediction Start')
    ax.set_xlabel("Hours Ago")
    ax.set_ylabel("Scaled Bike Count")
    ax.set_title("Historical Bike Demand (Input Sequence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col4:
    st.write("**Weather Features**")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    feature_names = ["Temperature", "Humidity", "Wind Speed"]
    feature_indices = [1, 3, 4]  # t1, hum, wind_speed
    
    for i, (name, idx) in enumerate(zip(feature_names, feature_indices)):
        if i < 3:
            row, col = i // 2, i % 2
            axes[row, col].plot(range(window_size), x_numeric[:, idx].numpy(), 'g-', linewidth=2)
            axes[row, col].set_title(f"{name} (Last {window_size}h)")
            axes[row, col].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    st.pyplot(fig)

# Show prediction details
st.subheader("Detailed Prediction Results")

pred_df = pd.DataFrame({
    'Hour': range(1, pred_horizon + 1),
    'True Value': y_true,
    'Predicted Value': y_pred,
    'Absolute Error': np.abs(y_true - y_pred),
    'Relative Error (%)': np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
})

st.dataframe(pred_df, use_container_width=True)

# Model comparison section
st.markdown("---")
st.subheader("Model Comparison")
st.write("""
**Transformer vs LSTM for Time Series Forecasting:**

**Transformer Advantages:**
- ✅ Better at capturing long-range dependencies through self-attention
- ✅ Parallel processing of sequences (faster training)
- ✅ Can handle variable-length sequences more naturally
- ✅ Attention mechanism provides interpretability
- ✅ No recurrent connections - less prone to vanishing gradients

**LSTM Advantages:**
- ✅ Sequential processing more intuitive for time series
- ✅ Fewer parameters for similar performance
- ✅ Better suited for very long sequences (memory efficiency)
- ✅ Inherent temporal ordering preservation

**Current Model Performance:**
- RMSE: {:.4f}
- MAE: {:.4f}
- MAPE: {:.2f}%
""".format(rmse_val, mae_val, mape_val))

if st.button("🔄 Generate New Prediction"):
    st.rerun()