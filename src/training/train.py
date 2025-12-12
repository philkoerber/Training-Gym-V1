"""Training script for time-series forecasting."""

import os
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from alpaca.data.timeframe import TimeFrame

from .callbacks import QuantConnectUploadCallback
from ..data.loader import load_all_symbols
from ..data.dataset import create_dataloaders
from ..models.patch_tst import TimeSeriesLightningModule

# ============================================================================
# TRAINING CONFIGURATION - Edit these values to change training parameters
# ============================================================================

# Data configuration
DAYS = 1460  # Days of historical data (4 years)
SEQ_LEN = 60  # Input sequence length (lookback window)
PRED_LEN = 1  # Prediction horizon

# Model configuration
D_MODEL = 64  # Transformer hidden dimension
NHEAD = 4  # Number of attention heads (must divide d_model)
NUM_LAYERS = 2  # Number of transformer layers
PATCH_LEN = 8  # Length of each patch
STRIDE = 4  # Stride between patches

# Training configuration
BATCH_SIZE = 32  # Training batch size
MAX_EPOCHS = 20  # Maximum training epochs
LEARNING_RATE = 1e-3  # Learning rate
NUM_WORKERS = 7  # Number of DataLoader worker processes (for cloud training)

# ============================================================================


def train(use_cloud: bool = True):
    """
    Train a time-series forecasting model on BTC/USD, ETH/USD, and LTC/USD.
    
    Args:
        use_cloud: If True, train on cloud/remote GPU servers; if False, use MPS on local Mac
    """
    # Set number of workers based on environment
    num_workers = NUM_WORKERS if use_cloud else 1
    print(f"Using {num_workers} DataLoader worker(s) ({'cloud' if use_cloud else 'local'})")
    
    # Load all three symbols with interconnectivity features
    timeframe_str = "minutely"
    print(f"\n{'='*50}")
    print(f"Loading BTC/USD, ETH/USD, LTC/USD")
    print(f"{timeframe_str} data ({DAYS} days)...")
    print(f"{'='*50}")
    
    df = load_all_symbols(
        days=DAYS,
        timeframe="minutely",
    )
    
    # Create dataloaders
    print(f"\nCreating dataloaders (seq_len={SEQ_LEN}, pred_len={PRED_LEN})...")
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        df,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get input dimension from dataset
    input_dim = len(train_ds.feature_cols)
    print(f"Input features: ({input_dim})")
    
    # Create model
    print(f"\n{'='*50}")
    print("Creating PatchTST model...")
    print(f"{'='*50}")
    
    model = TimeSeriesLightningModule(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN,
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        lr=LEARNING_RATE,
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="patchtst-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        QuantConnectUploadCallback(key_prefix="models", train_dataset=train_ds),
    ]
    
    # Logger
    logger = TensorBoardLogger("logs", name="patchtst")
    
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
        max_epochs=MAX_EPOCHS,
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
    parser.add_argument("--local", action="store_true", help="Train on local Mac with MPS instead of cloud GPU")
    
    args = parser.parse_args()
    
    # Default to cloud (use_cloud=True), unless --local flag is set
    train(use_cloud=not args.local)

