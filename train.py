"""Training script for time-series forecasting."""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data_loader import load_or_download
from dataset import create_dataloaders
from model import TimeSeriesLightningModule


def train(
    symbol: str = "BTC/USD",
    days: int = 60,
    seq_len: int = 24,
    pred_len: int = 1,
    model_type: str = "lstm",
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Train a time-series forecasting model.
    
    Args:
        symbol: Crypto pair to train on
        days: Days of historical data
        seq_len: Input sequence length (lookback)
        pred_len: Prediction horizon
        model_type: "lstm" or "transformer"
        hidden_dim: Model hidden dimension
        num_layers: Number of layers
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        lr: Learning rate
    """
    # Load data
    print(f"\n{'='*50}")
    print(f"Loading {symbol} data...")
    print(f"{'='*50}")
    df = load_or_download(symbol=symbol, days=days)
    
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
    ]
    
    # Logger
    logger = TensorBoardLogger("logs", name=model_type)
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
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
    
    parser = argparse.ArgumentParser(description="Train time-series forecaster")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train(
        symbol=args.symbol,
        days=args.days,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
    )

