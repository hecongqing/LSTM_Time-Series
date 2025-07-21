import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based cyclical features and return extended DataFrame."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # basic time parts
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # cyclical encoding for hour and month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


class BikeDataset(Dataset):
    """Pytorch Dataset for sliding window time series."""

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 24,
        pred_horizon: int = 6,
    ):
        """
        Args:
            data: 2D numpy array after scaling (num_samples, num_features).
            window_size: number of past timesteps fed into model.
            pred_horizon: number of future steps to predict.
        """
        self.X = []
        self.y = []
        for i in range(len(data) - window_size - pred_horizon + 1):
            self.X.append(data[i : i + window_size])
            # predict cnt column located at index 0 (assumed after scaling) for horizon steps
            self.y.append(
                data[i + window_size : i + window_size + pred_horizon, 0]
            )
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_process(
    csv_path: str,
    window_size: int = 24,
    pred_horizon: int = 6,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Load csv, engineer features, split, scale, and return datasets & scaler."""
    df = pd.read_csv(csv_path)
    df = create_features(df)

    # Select features (cnt + others)
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

    data = df[feature_cols].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    num_total = len(data_scaled)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    train_data = data_scaled[:num_train]
    val_data = data_scaled[num_train - window_size - pred_horizon + 1 : num_train + num_val]
    test_data = data_scaled[num_train + num_val - window_size - pred_horizon + 1 :]

    train_ds = BikeDataset(train_data, window_size, pred_horizon)
    val_ds = BikeDataset(val_data, window_size, pred_horizon)
    test_ds = BikeDataset(test_data, window_size, pred_horizon)

    return train_ds, val_ds, test_ds, scaler, len(feature_cols)