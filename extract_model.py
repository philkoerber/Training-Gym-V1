"""Backward compatibility wrapper for model extraction script.

This script imports from the new modular structure and maintains
backward compatibility with existing workflows.
"""

from src.utils.extract import extract_artifacts
import argparse
from alpaca.data.timeframe import TimeFrame

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
