from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WINDOW_SIZE = 24  # must match training/convert settings
INPUT_DIM = 13   # number of input features


class InputPayload(BaseModel):
    """Schema for prediction requests.

    Attributes
    ----------
    data : List[List[float]]
        2-D list with shape (WINDOW_SIZE, INPUT_DIM) representing a single
        time-series window that the model will use to forecast the next
        `pred_horizon` timesteps of bike demand (scaled values).
    """

    data: List[List[float]]


# Instantiate FastAPI app
app = FastAPI(
    title="London Bike Demand – ONNX Inference API",
    description="Serve the LSTM predictor via ONNX Runtime and FastAPI.",
    version="1.0.0",
)

# Load ONNX model at startup
ONNX_PATH = Path("outputs/model_best.onnx")
if not ONNX_PATH.exists():
    raise RuntimeError(
        f"ONNX model not found at {ONNX_PATH}. Run deploy/convert_to_onnx.py first."
    )

# Use CPUExecutionProvider by default; add others if available (e.g., CUDA, TensorRT)
session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


@app.post("/predict")
def predict(payload: InputPayload):
    """Return model predictions for a single input window.

    The response is a list of floats with length equal to the prediction horizon
    used during training (default 6).
    """

    # Convert input to NumPy array
    x_np = np.asarray(payload.data, dtype=np.float32)
    if x_np.ndim != 2:
        raise HTTPException(status_code=400, detail="Input must be 2-D list/array.")
    if x_np.shape != (WINDOW_SIZE, INPUT_DIM):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape ({WINDOW_SIZE}, {INPUT_DIM}); got {x_np.shape}",
        )

    # Add batch dimension
    x_np = np.expand_dims(x_np, axis=0)

    # Run inference
    preds = session.run(None, {"input": x_np})[0]  # shape: (1, pred_horizon)

    return {"prediction": preds.squeeze(0).tolist()}