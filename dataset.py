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
        
        # Always compute returns as target
        data = data.copy()
        # Compute forward returns: (next_close - current_close) / current_close
        if "close" not in data.columns:
            raise ValueError("Cannot compute returns: 'close' column not found")
        # Shift to get next close, then compute return
        data["target_returns"] = (data["close"].shift(-pred_len) - data["close"]) / data["close"]
        # Drop rows where returns are NaN (last pred_len rows)
        # This is safe because __len__ already accounts for pred_len
        data = data.dropna(subset=["target_returns"])
        # Use returns as target column
        target_col = "target_returns"
        
        self.target_col = target_col
        
        # Select features (excluding target column from input to avoid data leak)
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            feature_cols = list(feature_cols)  # Make a copy
        
        # Ensure target column exists in data (but we'll exclude it from input features)
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data columns: {list(data.columns)}")
        
        # Remove target from input features to avoid data leak
        # But keep it in a separate list for normalization and target extraction
        input_feature_cols = [col for col in feature_cols if col != target_col]
        
        # For normalization and target extraction, we need both input features AND target
        # This allows us to normalize them together (for consistent scaling) but exclude target from input
        all_cols_for_scaling = input_feature_cols + [target_col]
        
        # Store both: input features (for model) and all columns (for scaling)
        self.feature_cols = input_feature_cols  # What goes into the model
        self.all_cols_for_scaling = all_cols_for_scaling  # What we normalize (includes target)
        self.target_idx = len(input_feature_cols)  # Target is the last column in scaling array
        
        # Normalize: fit if scaler=None, transform if scaler provided
        # We normalize input features + target together, but only use input features in model
        if scaler is None:
            # Training: fit new scaler on all columns (input + target)
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data[all_cols_for_scaling].values)
        else:
            # Validation/Test: use provided scaler (must be already fitted)
            if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                raise ValueError("Provided scaler must be fitted before use")
            expected_cols = len(all_cols_for_scaling)
            if len(scaler.mean_) != expected_cols:
                raise ValueError(
                    f"Scaler was fitted on {len(scaler.mean_)} features, "
                    f"but data has {expected_cols} features (input + target). "
                    f"Feature columns must match exactly."
                )
            self.scaler = scaler
            data_scaled = self.scaler.transform(data[all_cols_for_scaling].values)
        
        # Store only input features in self.data (exclude target column)
        self.data = data_scaled[:, :len(input_feature_cols)]
        # Store target separately for __getitem__
        self.target_data = data_scaled[:, self.target_idx:self.target_idx+1]  # Keep as 2D for consistency
        
        # Store target scaler params for inverse transform
        # These are computed from returns
        self.target_mean = self.scaler.mean_[self.target_idx]
        self.target_std = self.scaler.scale_[self.target_idx]
        
        # Store original close column index for inference conversion (in input features)
        if "close" in self.feature_cols:
            self.close_idx = self.feature_cols.index("close")
        else:
            self.close_idx = None
    
    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]  # Input features only (no target column)
        # Extract target from separate target_data array
        y_start = idx + self.seq_len
        y_end = y_start + self.pred_len
        y = self.target_data[y_start:y_end, 0]  # Flatten to 1D
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
    
    # If using returns, target_col will be changed to "target_returns" inside TimeSeriesDataset
    # So we don't need to check for target_col in feature_cols here
    # The dataset will compute target_returns and handle validation internally
    
    # Verify all feature columns exist in all splits (but target may be computed dynamically)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Check base columns (excluding target_returns which will be computed)
        base_cols = [col for col in feature_cols if col != "target_returns"]
        missing_cols = set(base_cols) - set(split_df.columns)
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

