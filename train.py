"""Training script for time-series forecasting."""

import os
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from alpaca.data.timeframe import TimeFrame

from callbacks import QuantConnectUploadCallback
from data_loader import load_all_symbols
from dataset import create_dataloaders
from model import TimeSeriesLightningModule


def train(
    days: int = 1460,  # 4 years
    seq_len: int = 60,  # 60 minutes = 1 hour for minutely data
    pred_len: int = 1,
    model_type: str = "lstm",
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 1e-3,
    timeframe: TimeFrame = TimeFrame.Minute,
    use_cloud: bool = False,
):
    """
    Train a time-series forecasting model on BTC/USD, ETH/USD, and LTC/USD.
    
    Args:
        days: Days of historical data (default: 1460 = 4 years)
        seq_len: Input sequence length (lookback)
        pred_len: Prediction horizon
        model_type: "lstm" or "transformer"
        hidden_dim: Model hidden dimension
        num_layers: Number of layers
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        lr: Learning rate
        timeframe: Data granularity (TimeFrame.Minute or TimeFrame.Hour)
        use_cloud: If True, train on cloud/remote GPU servers; if False, use MPS on local Mac
    """
    # Load all three symbols with interconnectivity features
    timeframe_str = "minutely" if timeframe == TimeFrame.Minute else "hourly"
    print(f"\n{'='*50}")
    print(f"Loading BTC/USD, ETH/USD, LTC/USD")
    print(f"{timeframe_str} data ({days} days)...")
    print(f"{'='*50}")
    
    df = load_all_symbols(
        days=days,
        timeframe=timeframe,
    )
    
    # Create dataloaders
    print(f"\nCreating dataloaders (seq_len={seq_len}, pred_len={pred_len})...")
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        df,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get input dimension from dataset
    input_dim = len(train_ds.feature_cols)
    print(f"Input features ({input_dim}): {train_ds.feature_cols}")
    
    # Create model
    print(f"\n{'='*50}")
    print(f"Creating {model_type.upper()} model...")
    print(f"{'='*50}")
    
    model = TimeSeriesLightningModule(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pred_len=pred_len,
        seq_len=seq_len,
        lr=lr,
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{model_type}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        QuantConnectUploadCallback(key_prefix="models"),
    ]
    
    # Logger
    logger = TensorBoardLogger("logs", name=model_type)
    
    # Configure accelerator based on use_cloud flag
    if use_cloud:
        # For Lightning cloud/remote GPU servers
        # Check for Lightning credentials if using Lightning AI cloud
        lightning_api_key = os.getenv("LIGHTNING_API_KEY")
        if lightning_api_key:
            print(f"\n{'='*50}")
            print("Lightning AI cloud credentials detected")
            print(f"{'='*50}")
        else:
            print(f"\n{'='*50}")
            print("Training on remote GPU servers (Lightning AI credentials optional)")
            print(f"{'='*50}")
        
        accelerator = "gpu"
        devices = "auto"  # Auto-detect available GPUs
    else:
        # For local Mac with MPS
        print(f"\n{'='*50}")
        print("Training on local Mac with MPS (Metal)...")
        print(f"{'='*50}")
        accelerator = "mps"
        devices = 1
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
    )
    
    # Train
    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"{'='*50}")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print(f"\n{'='*50}")
    print("Testing...")
    print(f"{'='*50}")
    trainer.test(model, test_loader)
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train time-series forecaster on BTC/USD, ETH/USD, LTC/USD")
    parser.add_argument("--days", type=int, default=1460)  # 4 years
    parser.add_argument("--seq_len", type=int, default=60)  # 60 minutes for minutely data
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timeframe", type=str, default="minute", choices=["minute", "hour"])
    parser.add_argument("--cloud", action="store_true", help="Train on cloud/remote GPU servers instead of local MPS")
    
    args = parser.parse_args()
    
    # Convert timeframe string to TimeFrame enum
    tf = TimeFrame.Minute if args.timeframe == "minute" else TimeFrame.Hour
    
    train(
        days=args.days,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        timeframe=tf,
        use_cloud=args.cloud,
    )

