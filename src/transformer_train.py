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
from transformer_dataset import load_and_process_transformer
from transformer_model import SimplifiedTransformerBikePredictor, TransformerBikePredictor, RMSLELoss
from utils import rmse, mae, plot_loss, plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for London Bike Sharing demand prediction")
    parser.add_argument("--data-path", type=str, required=True, help="Path to london_merged.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--pred-horizon", type=int, default=6)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save-dir", type=str, default="transformer_outputs")
    parser.add_argument("--use-categorical", action="store_true", help="Use categorical features")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "rmsle"])
    parser.add_argument("--time-shuffle", action="store_true", help="Shuffle time sequences")
    args = parser.parse_args()
    return args


def custom_collate_fn(batch):
    """Custom collate function to handle None values in batch."""
    x_numeric = torch.stack([item[0] for item in batch])
    
    # Handle categorical features (may be None)
    x_categorical = None
    if batch[0][1] is not None:
        x_categorical = torch.stack([item[1] for item in batch])
    
    # Handle static features (may be None)  
    x_static = None
    if batch[0][2] is not None:
        x_static = torch.stack([item[2] for item in batch])
    
    y = torch.stack([item[3] for item in batch])
    
    return x_numeric, x_categorical, x_static, y


def train_model(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loading
    (train_ds, val_ds, test_ds, scaler, 
     num_numeric_features, categorical_dims, static_dims) = load_and_process_transformer(
        args.data_path, 
        args.window_size, 
        args.pred_horizon,
        use_categorical=args.use_categorical
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Model selection
    if args.use_categorical and (categorical_dims or static_dims):
        print(f"Using full Transformer with categorical features")
        print(f"Categorical dims: {categorical_dims}, Static dims: {static_dims}")
        model = TransformerBikePredictor(
            numeric_features=num_numeric_features,
            categorical_features_dims=categorical_dims,
            static_features_dims=static_dims,
            embed_size=args.embed_size,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            output_dim=args.pred_horizon,
            dropout=args.dropout,
        ).to(args.device)
    else:
        print(f"Using simplified Transformer with only numeric features")
        model = SimplifiedTransformerBikePredictor(
            input_dim=num_numeric_features,
            embed_size=args.embed_size,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            output_dim=args.pred_horizon,
            dropout=args.dropout,
        ).to(args.device)

    # Loss function
    if args.loss_type == "rmsle":
        criterion = RMSLELoss()
    else:
        criterion = torch.nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for x_numeric, x_categorical, x_static, y in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            x_numeric = x_numeric.to(args.device)
            y = y.to(args.device)
            
            if x_categorical is not None:
                x_categorical = x_categorical.to(args.device)
            if x_static is not None:
                x_static = x_static.to(args.device)

            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, SimplifiedTransformerBikePredictor):
                preds = model(x_numeric)
            else:
                preds = model(x_numeric, x_categorical, x_static)
                
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1

        epoch_train_loss = running_loss / num_batches

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for x_numeric, x_categorical, x_static, y in val_loader:
                x_numeric = x_numeric.to(args.device)
                y = y.to(args.device)
                
                if x_categorical is not None:
                    x_categorical = x_categorical.to(args.device)
                if x_static is not None:
                    x_static = x_static.to(args.device)

                if isinstance(model, SimplifiedTransformerBikePredictor):
                    preds = model(x_numeric)
                else:
                    preds = model(x_numeric, x_categorical, x_static)
                    
                loss = criterion(preds, y)
                val_running_loss += loss.item()
                val_num_batches += 1

        epoch_val_loss = val_running_loss / val_num_batches

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}"
        )

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "transformer_model_best.pt"))

        scheduler.step()

    # Plot training curves
    plot_loss(train_losses, val_losses, os.path.join(args.save_dir, "transformer_loss_curve.png"))

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "transformer_model_best.pt")))

    # Test evaluation
    model.eval()
    preds_all, targets_all = [], []
    
    with torch.no_grad():
        for x_numeric, x_categorical, x_static, y in test_loader:
            x_numeric = x_numeric.to(args.device)
            
            if x_categorical is not None:
                x_categorical = x_categorical.to(args.device)
            if x_static is not None:
                x_static = x_static.to(args.device)

            if isinstance(model, SimplifiedTransformerBikePredictor):
                preds = model(x_numeric).cpu()
            else:
                preds = model(x_numeric, x_categorical, x_static).cpu()
                
            preds_all.append(preds)
            targets_all.append(y)

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    test_rmse = rmse(preds_all, targets_all)
    test_mae = mae(preds_all, targets_all)
    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    # Plot predictions for first test sample
    x_numeric_first, x_categorical_first, x_static_first, y_first = test_ds[0]
    
    with torch.no_grad():
        x_numeric_batch = torch.tensor(x_numeric_first).unsqueeze(0).to(args.device)
        
        if isinstance(model, SimplifiedTransformerBikePredictor):
            pred_seq = model(x_numeric_batch).cpu().numpy().flatten()
        else:
            x_categorical_batch = None
            x_static_batch = None
            
            if x_categorical_first is not None:
                x_categorical_batch = torch.tensor(x_categorical_first).unsqueeze(0).to(args.device)
            if x_static_first is not None:
                x_static_batch = torch.tensor(x_static_first).unsqueeze(0).to(args.device)
                
            pred_seq = model(x_numeric_batch, x_categorical_batch, x_static_batch).cpu().numpy().flatten()

    true_seq = y_first.numpy()
    plot_predictions(
        true_seq, pred_seq, 
        os.path.join(args.save_dir, "transformer_prediction_plot.png"), 
        args.pred_horizon
    )

    # Save scaler and model info
    joblib.dump(scaler, os.path.join(args.save_dir, "transformer_feature_scaler.pkl"))
    
    # Save model configuration
    model_config = {
        'model_type': 'simplified' if isinstance(model, SimplifiedTransformerBikePredictor) else 'full',
        'num_numeric_features': num_numeric_features,
        'categorical_dims': categorical_dims,
        'static_dims': static_dims,
        'embed_size': args.embed_size,
        'num_heads': args.num_heads,
        'num_blocks': args.num_blocks,
        'output_dim': args.pred_horizon,
        'dropout': args.dropout,
    }
    joblib.dump(model_config, os.path.join(args.save_dir, "transformer_model_config.pkl"))


if __name__ == "__main__":
    args = parse_args()
    train_model(args)