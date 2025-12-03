"""Inference script for making predictions."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_or_download
from dataset import TimeSeriesDataset
from model import TimeSeriesLightningModule


def load_model(checkpoint_path: str) -> TimeSeriesLightningModule:
    """Load a trained model from checkpoint."""
    model = TimeSeriesLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def predict(
    model: TimeSeriesLightningModule,
    df: pd.DataFrame,
    seq_len: int = 24,
    pred_len: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset.
    
    Returns:
        (timestamps, actual_values, predicted_values)
    """
    dataset = TimeSeriesDataset(df, seq_len=seq_len, pred_len=pred_len)
    
    actuals = []
    predictions = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0)  # Add batch dimension
            
            y_hat = model(x).squeeze().numpy()
            y_actual = y.numpy()
            
            # Inverse transform to original scale
            y_hat = dataset.inverse_transform_target(y_hat)
            y_actual = dataset.inverse_transform_target(y_actual)
            
            predictions.append(y_hat)
            actuals.append(y_actual)
    
    # Get timestamps for predictions
    timestamps = df.index[seq_len : seq_len + len(predictions)]
    
    return (
        np.array(timestamps),
        np.array(actuals).flatten(),
        np.array(predictions).flatten(),
    )


def plot_predictions(
    timestamps: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    title: str = "Price Predictions",
    save_path: str | None = None,
):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(timestamps, actuals, label="Actual", alpha=0.7)
    plt.plot(timestamps, predictions, label="Predicted", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.show()


def calculate_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Calculate prediction metrics."""
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Direction accuracy
    actual_dir = np.diff(actuals) > 0
    pred_dir = np.diff(predictions) > 0
    direction_acc = np.mean(actual_dir == pred_dir) * 100
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Direction Accuracy %": direction_acc,
    }


if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=1)
    
    args = parser.parse_args()
    
    # Find latest checkpoint if not specified
    if args.checkpoint is None:
        checkpoints = glob.glob("checkpoints/*.ckpt")
        if not checkpoints:
            print("No checkpoints found. Train a model first with: python train.py")
            exit(1)
        args.checkpoint = max(checkpoints, key=lambda x: x)
        print(f"Using latest checkpoint: {args.checkpoint}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    
    # Load data
    print(f"Loading {args.symbol} data...")
    df = load_or_download(args.symbol, days=args.days)
    
    # Split to use only test portion
    test_start = int(len(df) * 0.9)
    test_df = df.iloc[test_start:]
    
    # Predict
    print("Making predictions...")
    timestamps, actuals, predictions = predict(
        model, test_df, seq_len=args.seq_len, pred_len=args.pred_len
    )
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Plot
    plot_predictions(timestamps, actuals, predictions, title=f"{args.symbol} Predictions")

