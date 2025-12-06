"""Time-series dataset for PyTorch using sklearn for normalization."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series forecasting with proper normalization.
    
    Creates sliding windows of (input_sequence, target) pairs.
    If scaler=None: fits new scaler (for training).
    If scaler provided: uses it (for val/test).
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 24,
        pred_len: int = 1,
        target_col: str = "close",
        feature_cols: list[str] | None = None,
        scaler: StandardScaler | None = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        
        # Select features
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = feature_cols
        
        # Verify target column exists
        if target_col not in feature_cols:
            raise ValueError(f"Target column '{target_col}' not found in feature columns: {feature_cols}")
        self.target_idx = feature_cols.index(target_col)
        
        # Normalize: fit if scaler=None, transform if scaler provided
        if scaler is None:
            # Training: fit new scaler
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(data[feature_cols].values)
        else:
            # Validation/Test: use provided scaler (must be already fitted)
            if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                raise ValueError("Provided scaler must be fitted before use")
            if len(scaler.mean_) != len(feature_cols):
                raise ValueError(
                    f"Scaler was fitted on {len(scaler.mean_)} features, "
                    f"but data has {len(feature_cols)} features. "
                    f"Feature columns must match exactly."
                )
            self.scaler = scaler
            self.data = self.scaler.transform(data[feature_cols].values)
        
        # Store target scaler params for inverse transform
        self.target_mean = self.scaler.mean_[self.target_idx]
        self.target_std = self.scaler.scale_[self.target_idx]
    
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
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
    num_workers: int = 0,
    target_col: str = "close",
    feature_cols: list[str] | None = None,
) -> tuple:
    """
    Create train/val/test dataloaders with temporal split.
    Scaler fitted on training data only, applied to val/test.
    
    Args:
        df: Full DataFrame with temporal ordering
        seq_len: Input sequence length
        pred_len: Prediction horizon
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        num_workers: Number of DataLoader workers
        target_col: Name of target column
        feature_cols: List of feature columns to use. If None, auto-detect from training data.
    
    Returns:
        (train_loader, val_loader, test_loader, train_ds)
    """
    from torch.utils.data import DataLoader
    
    # Temporal split (no shuffling - preserves temporal order)
    n = len(df)
    train_df = df.iloc[:int(n * train_ratio)].copy()
    val_df = df.iloc[int(n * train_ratio):int(n * (train_ratio + val_ratio))].copy()
    test_df = df.iloc[int(n * (train_ratio + val_ratio)):].copy()
    
    # Determine feature columns from training data to ensure consistency
    if feature_cols is None:
        feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Verify target column exists
    if target_col not in feature_cols:
        raise ValueError(f"Target column '{target_col}' not found in feature columns: {feature_cols}")
    
    # Verify all feature columns exist in all splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing_cols = set(feature_cols) - set(split_df.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns in {split_name} split: {missing_cols}")
    
    # Fit scaler on training data only
    train_ds = TimeSeriesDataset(
        train_df, 
        seq_len=seq_len, 
        pred_len=pred_len,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    
    # Apply training scaler to val/test (no fitting)
    val_ds = TimeSeriesDataset(
        val_df, 
        seq_len=seq_len, 
        pred_len=pred_len,
        target_col=target_col,
        feature_cols=feature_cols,
        scaler=train_ds.scaler,  # Use training scaler
    )
    test_ds = TimeSeriesDataset(
        test_df, 
        seq_len=seq_len, 
        pred_len=pred_len,
        target_col=target_col,
        feature_cols=feature_cols,
        scaler=train_ds.scaler,  # Use training scaler
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_ds

