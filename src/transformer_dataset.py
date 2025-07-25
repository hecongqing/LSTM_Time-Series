import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
import torch


def create_features_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for Transformer model."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Basic time parts (categorical)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day"] = df["timestamp"].dt.day

    # Cyclical encoding for continuous time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


class TransformerBikeDataset(Dataset):
    """Dataset for Transformer model with support for categorical and numeric features."""

    def __init__(
        self,
        numeric_data: np.ndarray,
        categorical_data: np.ndarray = None,
        static_data: np.ndarray = None,
        window_size: int = 24,
        pred_horizon: int = 6,
        target_idx: int = 0,
    ):
        """
        Args:
            numeric_data: (num_samples, num_numeric_features) - scaled numeric features
            categorical_data: (num_samples, num_categorical_features) - encoded categorical features  
            static_data: (num_samples, num_static_features) - encoded static features
            window_size: number of past timesteps
            pred_horizon: number of future steps to predict
            target_idx: index of target variable in numeric_data
        """
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.target_idx = target_idx
        
        self.X_numeric = []
        self.X_categorical = []
        self.X_static = []
        self.y = []
        
        # Create sliding windows
        for i in range(len(numeric_data) - window_size - pred_horizon + 1):
            # Numeric features window
            self.X_numeric.append(numeric_data[i:i + window_size])
            
            # Categorical features window (if provided)
            if categorical_data is not None:
                self.X_categorical.append(categorical_data[i:i + window_size])
            
            # Static features (same for all timesteps in window)
            if static_data is not None:
                self.X_static.append(static_data[i])
            
            # Target values for prediction horizon
            self.y.append(
                numeric_data[i + window_size:i + window_size + pred_horizon, target_idx]
            )
        
        self.X_numeric = np.array(self.X_numeric, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        if categorical_data is not None:
            self.X_categorical = np.array(self.X_categorical, dtype=np.int64)
        else:
            self.X_categorical = None
            
        if static_data is not None:
            self.X_static = np.array(self.X_static, dtype=np.int64)
        else:
            self.X_static = None

    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx):
        result = [torch.tensor(self.X_numeric[idx])]
        
        if self.X_categorical is not None:
            result.append(torch.tensor(self.X_categorical[idx]))
        else:
            result.append(None)
            
        if self.X_static is not None:
            result.append(torch.tensor(self.X_static[idx]))
        else:
            result.append(None)
            
        result.append(torch.tensor(self.y[idx]))
        
        return tuple(result)


def divide_shuffle_transformer(df, div_num):
    """Divide dataframe into chunks and shuffle within each chunk."""
    space = df.shape[0] // div_num
    division = np.arange(0, df.shape[0], space)
    return pd.concat([
        df.iloc[division[i]:division[i] + space, :].sample(frac=1) 
        for i in range(len(division))
    ])


def create_time_blocks_transformer(time_length, window_size):
    """Create time block indices for training."""
    start_idx = np.random.randint(0, window_size - 1)
    end_idx = time_length - window_size - 16 - 1
    time_indices = np.arange(start_idx, end_idx + 1, window_size)[:-1]
    time_indices = np.append(time_indices, end_idx)
    return time_indices


def load_and_process_transformer(
    csv_path: str,
    window_size: int = 24,
    pred_horizon: int = 6,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_categorical: bool = True,
):
    """Load and process data for Transformer model."""
    df = pd.read_csv(csv_path)
    df = create_features_transformer(df)

    # Define feature columns
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
    ] if use_categorical else []
    
    static_cols = [
        "is_holiday",
        "is_weekend",
    ] if use_categorical else []

    # Process numeric features
    numeric_data = df[numeric_cols].values
    scaler = StandardScaler()
    numeric_data_scaled = scaler.fit_transform(numeric_data)

    # Process categorical features
    categorical_data_encoded = None
    categorical_dims = []
    if use_categorical and categorical_cols:
        categorical_data = df[categorical_cols].copy()
        categorical_data_encoded = np.zeros_like(categorical_data.values)
        
        for i, col in enumerate(categorical_cols):
            le = LabelEncoder()
            categorical_data_encoded[:, i] = le.fit_transform(categorical_data[col])
            categorical_dims.append(len(le.classes_))

    # Process static features
    static_data_encoded = None
    static_dims = []
    if use_categorical and static_cols:
        static_data = df[static_cols].copy()
        static_data_encoded = np.zeros_like(static_data.values)
        
        for i, col in enumerate(static_cols):
            le = LabelEncoder()
            static_data_encoded[:, i] = le.fit_transform(static_data[col])
            static_dims.append(len(le.classes_))

    # Split data
    num_total = len(numeric_data_scaled)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)

    # Create datasets
    train_numeric = numeric_data_scaled[:num_train]
    val_numeric = numeric_data_scaled[num_train - window_size - pred_horizon + 1:num_train + num_val]
    test_numeric = numeric_data_scaled[num_train + num_val - window_size - pred_horizon + 1:]

    train_categorical = categorical_data_encoded[:num_train] if categorical_data_encoded is not None else None
    val_categorical = categorical_data_encoded[num_train - window_size - pred_horizon + 1:num_train + num_val] if categorical_data_encoded is not None else None
    test_categorical = categorical_data_encoded[num_train + num_val - window_size - pred_horizon + 1:] if categorical_data_encoded is not None else None

    train_static = static_data_encoded[:num_train] if static_data_encoded is not None else None
    val_static = static_data_encoded[num_train - window_size - pred_horizon + 1:num_train + num_val] if static_data_encoded is not None else None
    test_static = static_data_encoded[num_train + num_val - window_size - pred_horizon + 1:] if static_data_encoded is not None else None

    train_ds = TransformerBikeDataset(
        train_numeric, train_categorical, train_static, window_size, pred_horizon
    )
    val_ds = TransformerBikeDataset(
        val_numeric, val_categorical, val_static, window_size, pred_horizon
    )
    test_ds = TransformerBikeDataset(
        test_numeric, test_categorical, test_static, window_size, pred_horizon
    )

    return (
        train_ds, 
        val_ds, 
        test_ds, 
        scaler, 
        len(numeric_cols), 
        categorical_dims, 
        static_dims
    )