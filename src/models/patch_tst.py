"""Lightning model for time-series forecasting with PatchTST."""

import lightning as L
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """PatchTST: Patch-based Transformer for time-series forecasting."""
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
        seq_len: int = 60,
        patch_len: int = 8,
        stride: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding: project patches to d_model
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )
    
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split sequence into patches.
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            (batch, num_patches, patch_len * input_dim)
        """
        batch_size = x.shape[0]
        patches = []
        
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # (batch, patch_len, input_dim)
            patch = patch.reshape(batch_size, -1)  # (batch, patch_len * input_dim)
            patches.append(patch)
        
        return torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        patches = self.create_patches(x)  # (batch, num_patches, patch_len * input_dim)
        
        # Embed patches
        x = self.patch_embedding(patches)  # (batch, num_patches, d_model)
        x = x + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, num_patches, d_model)
        
        # Use last patch for prediction (most recent information)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Predict
        out = self.head(x)  # (batch, pred_len)
        return out


class TimeSeriesLightningModule(L.LightningModule):
    """Lightning wrapper for PatchTST model."""
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        pred_len: int = 1,
        seq_len: int = 60,
        patch_len: int = 8,
        stride: int = 4,
        lr: float = 1e-3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = PatchTST(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pred_len=pred_len,
            dropout=dropout,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
        )
        
        self.criterion = nn.MSELoss()
        self.lr = lr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log(f"{stage}_loss", loss, prog_bar=True)
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

