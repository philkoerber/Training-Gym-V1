"""Time-series dataset for PyTorch."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series forecasting.
    
    Creates sliding windows of (input_sequence, target) pairs.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 24,
        pred_len: int = 1,
        target_col: str = "close",
        feature_cols: list[str] | None = None,
    ):
        """
        Args:
            data: DataFrame with time-series data
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of prediction horizon
            target_col: Column to predict
            feature_cols: Columns to use as features (default: all numeric)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        
        # Select features
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = feature_cols
        
        # Get target column index
        self.target_idx = feature_cols.index(target_col)
        
        # Normalize data
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data[feature_cols].values)
        
        # Store target scaler params for inverse transform
        self.target_mean = self.scaler.mean_[self.target_idx]
        self.target_std = self.scaler.scale_[self.target_idx]
        
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Input sequence: all features
        x = self.data[idx : idx + self.seq_len]
        
        # Target: future values of target column
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len,
            self.target_idx
        ]
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        return y * self.target_std + self.target_mean


def create_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 24,
    pred_len: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple:
    """
    Create train/val/test dataloaders with temporal split.
    
    Returns:
        (train_loader, val_loader, test_loader, dataset)
    """
    from torch.utils.data import DataLoader
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_ds = TimeSeriesDataset(train_df, seq_len, pred_len)
    val_ds = TimeSeriesDataset(val_df, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_df, seq_len, pred_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, train_ds

