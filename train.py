"""Backward compatibility wrapper for training script.

This script imports from the new modular structure and maintains
backward compatibility with existing workflows.
"""

from src.training.train import train

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train time-series forecaster on BTC/USD, ETH/USD, LTC/USD")
    parser.add_argument("--local", action="store_true", help="Train on local Mac with MPS instead of cloud GPU")
    
    args = parser.parse_args()
    
    # Default to cloud (use_cloud=True), unless --local flag is set
    train(use_cloud=not args.local)
