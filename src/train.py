import argparse
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import joblib
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import load_and_process
from model import LSTMBikePredictor
from utils import rmse, mae, plot_loss, plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM for London Bike Sharing demand prediction")
    parser.add_argument("--data-path", type=str, required=True, help="Path to london_merged.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--pred-horizon", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save-dir", type=str, default="outputs")
    args = parser.parse_args()
    return args


def train_model(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_ds, val_ds, test_ds, scaler, input_dim = load_and_process(
        args.data_path, args.window_size, args.pred_horizon
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = LSTMBikePredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=args.pred_horizon,
    ).to(args.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            X = X.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(args.device)
                y = y.to(args.device)
                preds = model(X)
                loss = criterion(preds, y)
                val_running_loss += loss.item() * X.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}"
        )

        # Save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))

    # Plot loss curve
    plot_loss(train_losses, val_losses, os.path.join(args.save_dir, "loss_curve.png"))

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "model_best.pt")))

    # Evaluate on test set
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(args.device)
            preds = model(X).cpu()
            preds_all.append(preds)
            targets_all.append(y)
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    test_rmse = rmse(preds_all, targets_all)
    test_mae = mae(preds_all, targets_all)
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # Plot predictions for first instance in test loader
    first_X, first_y = test_ds[0]
    with torch.no_grad():
        pred_seq = model(torch.tensor(first_X).unsqueeze(0).to(args.device)).cpu().numpy().flatten()
    true_seq = first_y
    plot_predictions(true_seq, pred_seq, os.path.join(args.save_dir, "prediction_plot.png"), args.pred_horizon)

    # Save scaler
    joblib.dump(scaler, os.path.join(args.save_dir, "feature_scaler.pkl"))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)