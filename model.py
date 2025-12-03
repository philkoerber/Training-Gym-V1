"""Lightning model for time-series forecasting."""

import lightning as L
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Simple LSTM for time-series forecasting."""
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_len),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Predict
        out = self.fc(last_hidden)  # (batch, pred_len)
        return out


class TransformerForecaster(nn.Module):
    """Simple Transformer encoder for time-series forecasting."""
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
        seq_len: int = 24,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding
        
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Use mean pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        out = self.fc(x)  # (batch, pred_len)
        return out


class TimeSeriesLightningModule(L.LightningModule):
    """Lightning wrapper for time-series models."""
    
    def __init__(
        self,
        model_type: str = "lstm",
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pred_len: int = 1,
        seq_len: int = 24,
        lr: float = 1e-3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == "lstm":
            self.model = LSTMForecaster(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                pred_len=pred_len,
                dropout=dropout,
            )
        elif model_type == "transformer":
            self.model = TransformerForecaster(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_layers=num_layers,
                pred_len=pred_len,
                dropout=dropout,
                seq_len=seq_len,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.criterion = nn.MSELoss()
        self.lr = lr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        
        # Also log MAE
        mae = torch.mean(torch.abs(y_hat - y))
        self.log(f"{stage}_mae", mae, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

