import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import requests

from dataset import load_and_process, BikeDataset, create_features
# from model import LSTMBikePredictor  # Local model inference no longer used
# from utils import rmse, mae  # Metrics computed directly with NumPy

st.set_page_config(page_title="London Bike Demand Predictor", layout="wide")

st.title("🚲 London Bike Sharing Demand Prediction")

st.markdown(
    """使用训练好的 PyTorch LSTM 模型，预测伦敦共享单车未来需求量，并可视化模型结构与预测效果。"""
)

# Sidebar configuration
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input(
    "Model weights (.pt)", value="outputs/model_best.pt", help="路径可修改为对应 horizon 训练好的模型权重"
)
scaler_path = st.sidebar.text_input(
    "Scaler pickle (.pkl)", value="outputs/feature_scaler.pkl"
)
data_path = st.sidebar.text_input(
    "Dataset CSV", value="data/raw/london_merged.csv"
)
pred_horizon = st.sidebar.selectbox("Prediction horizon", options=[1, 6], index=1)
window_size = st.sidebar.number_input("Window size (hours)", value=24, step=1)
# URL of the running FastAPI service delivering ONNX predictions
server_url = st.sidebar.text_input("Inference API URL", value="http://localhost:8000")

# No local device needed when using remote inference

# Load scaler
@st.cache_resource
def load_scaler(path: str):
    return joblib.load(path)

# Data loading
@st.cache_data
def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = create_features(df)
    return df

if Path(data_path).exists():
    df_full = prepare_data(data_path)
else:
    st.error("Dataset path invalid")
    st.stop()

feature_cols = [
    "cnt",
    "t1",
    "t2",
    "hum",
    "wind_speed",
    "weather_code",
    "is_holiday",
    "is_weekend",
    "season",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

scaler = load_scaler(scaler_path) if Path(scaler_path).exists() else None

if scaler is None:
    st.error("Scaler file not found, 先训练模型生成 scaler")
    st.stop()

scaled = scaler.transform(df_full[feature_cols])

# Build dataset for prediction (whole)
ds = BikeDataset(scaled, window_size, pred_horizon)

# Local model loading removed – predictions are obtained via remote API

st.sidebar.markdown("---")
# Build a mapping from dataset index to the corresponding timestamp (window end time)
window_end_timestamps = (
    df_full["timestamp"].iloc[window_size - 1 : len(df_full) - pred_horizon].dt.strftime("%Y-%m-%d %H:%M")
)
# Convert to list for deterministic order in the UI
timestamp_options = window_end_timestamps.tolist()

selected_time = st.sidebar.selectbox(
    "选择时间窗口（窗口结束时间）", options=timestamp_options, index=0
)
# Map the selected timestamp back to the dataset index
idx = timestamp_options.index(selected_time)

# X, y preparation remains the same
X, y_true = ds[idx]
# Send prediction request to the FastAPI ONNX inference server
try:
    response = requests.post(
        f"{server_url}/predict",
        json={"data": X.tolist()},
        timeout=10,
    )
    response.raise_for_status()
    y_pred = np.array(response.json()["prediction"])
except Exception as e:
    st.error(f"Inference request failed: {e}")
    st.stop()

y_true = y_true[: pred_horizon]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction vs. Ground Truth")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(range(pred_horizon), y_true, label="True")
    ax.plot(range(pred_horizon), y_pred, label="Predicted")
    ax.set_xlabel("Future Hour")
    ax.set_ylabel("Scaled cnt")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Metrics")
    rmse_val = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae_val = np.mean(np.abs(y_pred - y_true))
    st.metric("RMSE", f"{rmse_val:.4f}")
    st.metric("MAE", f"{mae_val:.4f}")

st.markdown("---")

# Removed local model architecture display because inference now happens remotely