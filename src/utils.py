import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


def plot_loss(train_losses: List[float], val_losses: List[float], save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions(
    true_seq: np.ndarray,
    pred_seq: np.ndarray,
    save_path: str,
    horizon: int = 6,
):
    plt.figure(figsize=(10, 5))
    plt.plot(true_seq[: 24 * 3], label="True")
    plt.plot(pred_seq[: 24 * 3], label="Predicted")
    plt.xlabel("Hours")
    plt.ylabel("Scaled cnt")
    plt.title(f"Next {horizon}h prediction vs. ground truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()