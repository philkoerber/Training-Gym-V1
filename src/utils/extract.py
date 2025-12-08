"""Extract model artifacts from a checkpoint for QuantConnect deployment.

This script extracts the PyTorch model, scaler, feature columns, and other
necessary artifacts from a Lightning checkpoint and saves them in a folder
ready for QuantConnect upload.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from ..data.loader import load_all_symbols
from ..data.dataset import create_dataloaders
from ..models.patch_tst import TimeSeriesLightningModule
from alpaca.data.timeframe import TimeFrame


def extract_artifacts(
    checkpoint_path: str,
    output_dir: str = None,
    days: int = 1460,
    timeframe: TimeFrame = TimeFrame.Hour,
):
    """
    Extract all model artifacts from a checkpoint.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint file
        output_dir: Directory to save artifacts (default: checkpoints/patchtst-{timestamp})
        days: Days of data to use for recreating the scaler (should match training)
        timeframe: Timeframe used during training
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    lightning_model = TimeSeriesLightningModule.load_from_checkpoint(str(checkpoint_path))
    pytorch_model = lightning_model.model
    pytorch_model.eval()
    
    hparams = lightning_model.hparams
    print(f"\nModel hyperparameters:")
    for key, value in hparams.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("checkpoints") / f"patchtst-{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving artifacts to {output_dir}...")
    
    # 1. Save PyTorch model state dict
    model_path = output_dir / "model.pt"
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'hyperparameters': dict(hparams),
    }, model_path)
    print(f"  ✓ Saved model to {model_path.name}")
    
    # 2. Recreate training dataset to get scaler and feature columns
    print(f"\nRecreating training dataset to extract scaler...")
    print(f"  Loading data (days={days}, timeframe={timeframe})...")
    
    df = load_all_symbols(days=days, timeframe=timeframe)
    
    print(f"  Creating dataloaders...")
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        df,
        seq_len=hparams['seq_len'],
        pred_len=hparams['pred_len'],
        batch_size=32,  # Doesn't matter for extraction
        num_workers=0,
    )
    
    # 3. Save scaler
    if hasattr(train_ds, 'scaler'):
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_ds.scaler, f)
        print(f"  ✓ Saved scaler to {scaler_path.name}")
    else:
        print("  ⚠ Warning: Scaler not found in dataset")
    
    # 4. Save feature columns
    if hasattr(train_ds, 'feature_cols'):
        feature_cols_path = output_dir / "feature_columns.pkl"
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(train_ds.feature_cols, f)
        print(f"  ✓ Saved feature columns to {feature_cols_path.name}")
        print(f"    Total features: {len(train_ds.feature_cols)}")
    else:
        print("  ⚠ Warning: Feature columns not found in dataset")
    
    # 5. Save target normalization parameters
    if hasattr(train_ds, 'target_mean') and hasattr(train_ds, 'target_std'):
        target_stats = {
            'target_mean': train_ds.target_mean,
            'target_std': train_ds.target_std,
            'target_idx': train_ds.target_idx,
        }
        target_stats_path = output_dir / "target_stats.pkl"
        with open(target_stats_path, 'wb') as f:
            pickle.dump(target_stats, f)
        print(f"  ✓ Saved target stats to {target_stats_path.name}")
        print(f"    Target mean: {train_ds.target_mean:.6f}")
        print(f"    Target std: {train_ds.target_std:.6f}")
    else:
        print("  ⚠ Warning: Target stats not found in dataset")
    
    # 6. Save model architecture info
    model_info = {
        'model_type': 'PatchTST',
        'hyperparameters': dict(hparams),
        'extraction_timestamp': datetime.now().isoformat(),
        'checkpoint_path': str(checkpoint_path),
    }
    info_path = output_dir / "model_info.pkl"
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"  ✓ Saved model info to {info_path.name}")
    
    print(f"\n{'='*50}")
    print(f"✓ Extraction complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"    - {file.name} ({size_kb:.2f} KB)")
    print(f"{'='*50}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model artifacts from a Lightning checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint file (e.g., checkpoints/patchtst-epoch=23-val_loss=0.0608.ckpt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoints/patchtst-{timestamp})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1460,
        help="Days of data to use for recreating scaler (default: 1460)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        choices=["Hour", "Minute"],
        default="Hour",
        help="Timeframe used during training (default: Hour)",
    )
    
    args = parser.parse_args()
    
    timeframe = TimeFrame.Hour if args.timeframe == "Hour" else TimeFrame.Minute
    
    extract_artifacts(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        days=args.days,
        timeframe=timeframe,
    )

