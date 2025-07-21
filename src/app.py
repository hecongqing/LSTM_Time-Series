import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from dataset import load_and_process, BikeDataset, create_features
from model import LSTMBikePredictor
from utils import rmse, mae

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

device = "cpu"

# Load model & scaler
@st.cache_resource
def load_model(model_path: str, input_dim: int, horizon: int):
    model = LSTMBikePredictor(input_dim=input_dim, output_dim=horizon)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

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

model = load_model(model_path, input_dim=scaled.shape[1], horizon=pred_horizon)

st.sidebar.markdown("---")
idx = st.sidebar.slider("Select sample index", min_value=0, max_value=len(ds) - 1, value=0)

with torch.no_grad():
    X, y_true = ds[idx]
    X_tensor = torch.tensor(X).unsqueeze(0).to(device)
    y_pred = model(X_tensor).cpu().numpy().flatten()

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

st.subheader("Model Architecture")
st.text(str(model))