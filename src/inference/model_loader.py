"""
Model loader for QuantConnect deployment.

This module provides a clean interface to load all artifacts from a trained model
and use them for inference in live trading.

================================================================================
HOW TO USE THIS LOADER IN QUANTCONNECT
================================================================================

1. UPLOAD FILES TO QUANTCONNECT OBJECT STORE
   -----------------------------------------
   Upload the entire checkpoint folder to QuantConnect's Object Store:
   - model.pt
   - scaler.pkl
   - feature_columns.pkl
   - target_stats.pkl
   - model_info.pkl (optional)
   
   Also upload this file (model_loader.py) and feature_engineering.py


2. IN YOUR QUANTCONNECT ALGORITHM
   -------------------------------
   
   from AlgorithmImports import *
   import pickle
   import torch
   from model_loader import TradingModelLoader
   from feature_engineering import engineer_features
   
   class MyTradingAlgorithm(QCAlgorithm):
       
       def Initialize(self):
           self.SetStartDate(2024, 1, 1)
           self.SetCash(100000)
           
           # Add crypto assets
           self.btc = self.AddCrypto("BTCUSD", Resolution.Hour).Symbol
           
           # Load the trained model from Object Store
           # Option A: If files are in a folder structure
           model_dir = self.ObjectStore.GetFilePath("models/patchtst-20251209-181235")
           self.model = TradingModelLoader.load(model_dir)
           
           # Option B: Load files individually (if Object Store doesn't support folders)
           # self.model = TradingModelLoader.load_from_object_store(self, "models/patchtst")
           
           # Store historical data for feature calculation
           self.lookback = 300  # Need enough history for features + seq_len
           
       def OnData(self, data):
           if not data.ContainsKey(self.btc):
               return
           
           # Get historical OHLCV data
           history = self.History(self.btc, self.lookback, Resolution.Hour)
           if len(history) < self.lookback:
               return
           
           # Convert to DataFrame format expected by the model
           df = history.loc[self.btc]
           df = df.rename(columns={
               'open': 'open', 
               'high': 'high', 
               'low': 'low', 
               'close': 'close',
               'volume': 'volume'
           })
           
           # Engineer features (same as training!)
           df = engineer_features(df, drop_na=True)
           
           # Make prediction
           try:
               prediction = self.model.predict(df)
               direction, confidence = self.model.predict_direction(df)
               
               # Trading logic
               current_price = data[self.btc].Close
               
               if direction == "up" and confidence > 0.02:
                   if not self.Portfolio[self.btc].Invested:
                       self.SetHoldings(self.btc, 0.5)
                       self.Debug(f"BUY signal: predicted={prediction[0]:.2f}, current={current_price:.2f}")
               
               elif direction == "down" and confidence > 0.02:
                   if self.Portfolio[self.btc].Invested:
                       self.Liquidate(self.btc)
                       self.Debug(f"SELL signal: predicted={prediction[0]:.2f}, current={current_price:.2f}")
                       
           except Exception as e:
               self.Debug(f"Prediction error: {e}")


3. REQUIRED DEPENDENCIES
   ----------------------
   QuantConnect supports PyTorch and scikit-learn out of the box.
   Make sure your algorithm imports torch and sklearn.


4. FEATURE ENGINEERING
   --------------------
   CRITICAL: You must apply the EXACT same feature engineering as during training!
   
   The feature columns expected by the model are stored in feature_columns.pkl.
   Use engineer_features() from feature_engineering.py to create them.
   
   Example feature columns (order matters!):
   ['open', 'high', 'low', 'close', 'volume', 'hour_sin', 'hour_cos', 
    'day_of_week_sin', 'day_of_week_cos', 'sma_7', 'sma_14', 'rsi', ...]


5. MULTI-SYMBOL PREDICTIONS
   -------------------------
   If your model was trained on multiple symbols with interconnectivity features,
   you need to compute those features on all symbols before prediction:
   
   from feature_engineering import engineer_features, add_interconnectivity_features
   
   # Get history for all symbols
   btc_df = self.History(self.btc, lookback, Resolution.Hour).loc[self.btc]
   eth_df = self.History(self.eth, lookback, Resolution.Hour).loc[self.eth]
   
   # Engineer individual features
   btc_df = engineer_features(btc_df, drop_na=False)
   eth_df = engineer_features(eth_df, drop_na=False)
   
   # Add interconnectivity features
   symbol_dfs = add_interconnectivity_features({
       'BTCUSD': btc_df,
       'ETHUSD': eth_df
   })
   
   # Now predict
   prediction = self.model.predict(symbol_dfs['BTCUSD'].dropna())


6. DEBUGGING TIPS
   ---------------
   - Check that df.columns matches self.model.feature_columns
   - Ensure you have at least seq_len rows after dropping NaN
   - Print self.model to see model configuration
   - Use predict(df, denormalize=False) to see raw model output

================================================================================
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelConfig:
    """Configuration loaded from saved model."""
    input_dim: int
    d_model: int
    nhead: int
    num_layers: int
    pred_len: int
    seq_len: int
    patch_len: int
    stride: int
    dropout: float = 0.1


class PatchTSTInference(nn.Module):
    """
    Minimal PatchTST model for inference only.
    
    Standalone version that doesn't require Lightning or training utilities.
    This can be used in QuantConnect without heavy dependencies.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.input_dim = config.input_dim
        self.num_patches = (config.seq_len - config.patch_len) // config.stride + 1
        
        # Build model architecture (must match training architecture exactly)
        self.patch_embedding = nn.Linear(config.patch_len * config.input_dim, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, config.d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_len),
        )
    
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split sequence into patches."""
        batch_size = x.shape[0]
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :].reshape(batch_size, -1)
            patches.append(patch)
        return torch.stack(patches, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, seq_len, input_dim) -> (batch, pred_len)"""
        patches = self.create_patches(x)
        x = self.patch_embedding(patches)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last patch for prediction
        return self.head(x)


class TradingModelLoader:
    """
    Main loader class for using trained models in QuantConnect.
    
    This class bundles everything needed for inference:
    - The PyTorch model (PatchTST)
    - The scaler for input normalization
    - Feature column names and ordering
    - Target statistics for denormalization
    
    Usage:
        # Load from directory
        model = TradingModelLoader.load("/path/to/checkpoint/")
        
        # Make predictions
        prediction = model.predict(ohlcv_dataframe)
        
        # Get direction signal
        direction, confidence = model.predict_direction(ohlcv_dataframe)
    """
    
    def __init__(
        self,
        pytorch_model: nn.Module,
        scaler: StandardScaler,
        feature_columns: list[str],
        target_mean: float,
        target_std: float,
        target_idx: int,
        config: ModelConfig,
        model_info: dict,
    ):
        self.model = pytorch_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_idx = target_idx
        self.config = config
        self.model_info = model_info
        
        # Pre-compute derived attributes
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_features = len(feature_columns)
        
        # Find close column index for inference conversion (always returns now)
        if "close" in feature_columns:
            self.close_idx = feature_columns.index("close")
        else:
            self.close_idx = None
        
        # Set model to evaluation mode
        self.model.eval()
    
    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "TradingModelLoader":
        """
        Load all model artifacts from a directory.
        
        Args:
            model_dir: Path to directory containing model artifacts
                       (model.pt, scaler.pkl, feature_columns.pkl, target_stats.pkl)
        
        Returns:
            TradingModelLoader instance ready for inference
        
        Raises:
            FileNotFoundError: If model directory or required files don't exist
            ValueError: If artifacts are incompatible
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Check required files exist
        required_files = ["model.pt", "scaler.pkl", "feature_columns.pkl", "target_stats.pkl"]
        for filename in required_files:
            if not (model_dir / filename).exists():
                raise FileNotFoundError(f"Required file not found: {model_dir / filename}")
        
        # 1. Load model weights and hyperparameters
        model_path = model_dir / "model.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        hparams = checkpoint["hyperparameters"]
        config = ModelConfig(
            input_dim=hparams["input_dim"],
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            num_layers=hparams["num_layers"],
            pred_len=hparams["pred_len"],
            seq_len=hparams["seq_len"],
            patch_len=hparams["patch_len"],
            stride=hparams["stride"],
            dropout=hparams.get("dropout", 0.1),
        )
        
        # Create model and load weights
        model = PatchTSTInference(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Load scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        # 3. Load feature columns
        feature_cols_path = model_dir / "feature_columns.pkl"
        with open(feature_cols_path, "rb") as f:
            feature_columns = pickle.load(f)
        
        # 4. Load target stats
        target_stats_path = model_dir / "target_stats.pkl"
        with open(target_stats_path, "rb") as f:
            target_stats = pickle.load(f)
        
        # 5. Load model info (optional metadata)
        model_info_path = model_dir / "model_info.pkl"
        model_info = {}
        if model_info_path.exists():
            with open(model_info_path, "rb") as f:
                model_info = pickle.load(f)
        
        # Validate consistency
        if len(feature_columns) != config.input_dim:
            raise ValueError(
                f"Feature columns ({len(feature_columns)}) don't match "
                f"model input_dim ({config.input_dim})"
            )
        
        return cls(
            pytorch_model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            target_mean=target_stats["target_mean"],
            target_std=target_stats["target_std"],
            target_idx=target_stats["target_idx"],
            config=config,
            model_info=model_info,
        )
    
    @classmethod
    def load_from_files(
        cls,
        model_pt_path: str,
        scaler_path: str,
        feature_cols_path: str,
        target_stats_path: str,
        model_info_path: Optional[str] = None,
    ) -> "TradingModelLoader":
        """
        Load model from individual file paths.
        
        Useful when files are not in a single directory (e.g., Object Store).
        
        Args:
            model_pt_path: Path to model.pt
            scaler_path: Path to scaler.pkl
            feature_cols_path: Path to feature_columns.pkl
            target_stats_path: Path to target_stats.pkl
            model_info_path: Optional path to model_info.pkl
        
        Returns:
            TradingModelLoader instance ready for inference
        """
        # 1. Load model weights and hyperparameters
        checkpoint = torch.load(model_pt_path, map_location="cpu", weights_only=False)
        
        hparams = checkpoint["hyperparameters"]
        config = ModelConfig(
            input_dim=hparams["input_dim"],
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            num_layers=hparams["num_layers"],
            pred_len=hparams["pred_len"],
            seq_len=hparams["seq_len"],
            patch_len=hparams["patch_len"],
            stride=hparams["stride"],
            dropout=hparams.get("dropout", 0.1),
        )
        
        model = PatchTSTInference(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Load scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        # 3. Load feature columns
        with open(feature_cols_path, "rb") as f:
            feature_columns = pickle.load(f)
        
        # 4. Load target stats
        with open(target_stats_path, "rb") as f:
            target_stats = pickle.load(f)
        
        # 5. Load model info (optional)
        model_info = {}
        if model_info_path:
            with open(model_info_path, "rb") as f:
                model_info = pickle.load(f)
        
        return cls(
            pytorch_model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            target_mean=target_stats["target_mean"],
            target_std=target_stats["target_std"],
            target_idx=target_stats["target_idx"],
            config=config,
            model_info=model_info,
        )
    
    def preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess raw OHLCV data into model input.
        
        Args:
            df: DataFrame with at least seq_len rows containing the feature columns.
                Must have the same feature columns as used during training.
        
        Returns:
            Tensor of shape (1, seq_len, n_features) ready for model input
        
        Raises:
            ValueError: If missing columns or insufficient data
        """
        # Ensure we have exactly the columns the model expects
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns: {missing_cols}\n"
                f"Expected columns: {self.feature_columns}\n"
                f"Got columns: {list(df.columns)}"
            )
        
        # Select and order columns (order matters!)
        data = df[self.feature_columns].values
        
        # Take the last seq_len rows
        if len(data) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows of data, got {len(data)}. "
                f"Make sure to collect enough historical data."
            )
        data = data[-self.seq_len:]
        
        # Check for NaN values
        if np.isnan(data).any():
            nan_cols = [
                self.feature_columns[i] 
                for i in range(len(self.feature_columns)) 
                if np.isnan(data[:, i]).any()
            ]
            raise ValueError(
                f"NaN values found in columns: {nan_cols}. "
                f"Make sure feature engineering completed properly."
            )
        
        # Normalize using the training scaler
        data_scaled = self.scaler.transform(data)
        
        # Convert to tensor: (1, seq_len, n_features)
        tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
        return tensor
    
    def predict(
        self, 
        df: pd.DataFrame, 
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Make a prediction on OHLCV data.
        
        Args:
            df: DataFrame with feature columns (needs at least seq_len rows).
                Must contain all columns in self.feature_columns.
            denormalize: If True, return prediction in original price scale.
                        If False, return normalized prediction.
        
        Returns:
            Numpy array of predicted values (shape: (pred_len,))
        """
        # Preprocess input
        x = self.preprocess(df)
        
        # Run inference (no gradient computation needed)
        with torch.no_grad():
            y_pred = self.model(x)
        
        prediction = y_pred.squeeze().numpy()
        
        # Handle single prediction case
        if prediction.ndim == 0:
            prediction = np.array([prediction])
        
        # Denormalize if requested
        if denormalize:
            # Always denormalize returns and convert to prices
            # Denormalize returns
            prediction_returns = prediction * self.target_std + self.target_mean
            # Convert returns to prices: predicted_price = current_price * (1 + return)
            # Get current close price directly from dataframe (raw value, not normalized)
            if "close" in df.columns:
                current_close = df["close"].iloc[-1]
            elif self.close_idx is not None:
                # Fallback: get from feature columns and denormalize
                current_data = df[self.feature_columns].iloc[-1:].values
                current_data_scaled = self.scaler.transform(current_data)
                current_close_scaled = current_data_scaled[0, self.close_idx]
                current_close = current_close_scaled * self.scaler.scale_[self.close_idx] + self.scaler.mean_[self.close_idx]
            else:
                # Can't convert to price, return returns instead
                return prediction_returns
            # Convert predicted returns to predicted prices
            prediction = current_close * (1 + prediction_returns)
        
        return prediction
    
    def predict_direction(
        self, 
        df: pd.DataFrame,
        threshold: float = 0.01
    ) -> tuple[str, float]:
        """
        Predict direction of price movement.
        
        Args:
            df: DataFrame with feature columns
            threshold: Minimum change to signal up/down.
                      For returns: threshold in return units (e.g., 0.01 = 1%).
                      For prices: threshold in normalized price units (default: 0.01)
        
        Returns:
            Tuple of (direction, confidence) where:
            - direction: 'up', 'down', or 'flat'
            - confidence: magnitude of predicted change
        """
        # Get prediction (normalized)
        pred_scaled = self.predict(df, denormalize=False)
        
        # Calculate predicted change
        if self.pred_len == 1:
            pred_value = pred_scaled[0]
        else:
            pred_value = pred_scaled[-1]  # Use last prediction horizon
        
        # Always denormalize return prediction
        pred_return = pred_value * self.target_std + self.target_mean
        # For returns, positive = up, negative = down
        pred_change = pred_return
        # Use threshold directly as return (e.g., 0.01 = 1%)
        
        # Convert to direction
        if pred_change > threshold:
            direction = "up"
        elif pred_change < -threshold:
            direction = "down"
        else:
            direction = "flat"
        
        return direction, float(abs(pred_change))
    
    def predict_with_context(self, df: pd.DataFrame) -> dict:
        """
        Make prediction with full context information.
        
        Args:
            df: DataFrame with feature columns
        
        Returns:
            Dictionary with prediction details:
            - 'prediction': denormalized predicted price(s)
            - 'prediction_normalized': normalized prediction
            - 'current_price': current close price
            - 'predicted_change': absolute price change
            - 'predicted_return': percentage return
            - 'direction': 'up', 'down', or 'flat'
            - 'confidence': magnitude of change (normalized)
        """
        # Current price (always use close column)
        if "close" in self.feature_columns:
            current_price = df["close"].iloc[-1]
        else:
            # Fallback to target column
            current_price = df[self.feature_columns[self.target_idx]].iloc[-1]
        
        # Predictions
        pred_denorm = self.predict(df, denormalize=True)
        pred_norm = self.predict(df, denormalize=False)
        
        # Direction
        direction, confidence = self.predict_direction(df)
        
        # Calculate changes
        if self.pred_len == 1:
            predicted_price = pred_denorm[0]
        else:
            predicted_price = pred_denorm[-1]
        
        # Prediction is always a price (converted from return)
        predicted_change = predicted_price - current_price
        predicted_return = (predicted_change / current_price) * 100
        
        return {
            "prediction": pred_denorm,
            "prediction_normalized": pred_norm,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change": predicted_change,
            "predicted_return": predicted_return,
            "direction": direction,
            "confidence": confidence,
        }
    
    def get_required_history_length(self) -> int:
        """
        Get the minimum number of rows needed for prediction.
        
        This accounts for:
        - Model sequence length
        - Feature engineering warmup (moving averages, etc.)
        
        Returns:
            Recommended minimum rows of historical data
        """
        # seq_len for model + 200 for longest moving average + buffer
        return self.seq_len + 250
    
    def __repr__(self) -> str:
        return (
            f"TradingModelLoader(\n"
            f"  model_type={self.model_info.get('model_type', 'PatchTST')},\n"
            f"  seq_len={self.seq_len},\n"
            f"  pred_len={self.pred_len},\n"
            f"  n_features={self.n_features},\n"
            f"  d_model={self.config.d_model},\n"
            f"  feature_columns={self.feature_columns[:5]}{'...' if len(self.feature_columns) > 5 else ''}\n"
            f")"
        )

