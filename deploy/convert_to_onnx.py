#!/usr/bin/env python
"""Convert trained LSTM bike–demand predictor to ONNX format.

Usage:
    python deploy/convert_to_onnx.py \
        --model-path outputs/model_best.pt \
        --onnx-path outputs/model_best.onnx \
        --input-dim 13 --window-size 24 --pred-horizon 6

This script loads the PyTorch model checkpoint and exports it to ONNX so that
it can be served efficiently by onnxruntime (or TensorRT, OpenVINO, etc.).
"""

import argparse
from pathlib import Path
import sys

import torch

# Add src directory to import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from model import LSTMBikePredictor  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="Export LSTM model to ONNX format")
    parser.add_argument("--model-path", type=str, default="outputs/model_best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--onnx-path", type=str, default="outputs/model_best.onnx", help="Output path for .onnx file")
    parser.add_argument("--input-dim", type=int, default=13, help="Feature dimension used during training")
    parser.add_argument("--window-size", type=int, default=24, help="Sequence length (number of past timesteps)")
    parser.add_argument("--pred-horizon", type=int, default=6, help="Number of future steps predicted by the model")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension; must match training config")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers; must match training config")
    return parser.parse_args()


def main():
    args = get_args()

    # Instantiate model
    model = LSTMBikePredictor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=args.pred_horizon,
    )

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Dummy input of shape (batch, seq_len, input_dim)
    dummy_input = torch.randn(1, args.window_size, args.input_dim, dtype=torch.float32)

    # Export to ONNX
    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch"},
        },
    )

    print(f"ONNX model exported to {onnx_path.resolve()}")


if __name__ == "__main__":
    main()