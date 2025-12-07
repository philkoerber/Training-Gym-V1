# Training Gym V1

A lightweight time-series forecasting toolkit for crypto price prediction. Train PatchTST (Patch-based Transformer) models on multi-symbol crypto data with advanced feature engineering, then deploy to QuantConnect.

## Features

- **Models**: PatchTST (Patch-based Transformer) architecture for time-series forecasting
- **Multi-Symbol Support**: Train on BTC/USD, ETH/USD, and LTC/USD simultaneously with interconnectivity features
- **Feature Engineering**: 
  - Time-based features (hour, day of week, day of month, month) with sin/cos encoding
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, moving averages)
  - Cross-symbol features (price ratios, relative returns, correlations, spreads, relative strength)
- **Data**: Automatic download from Alpaca with local caching and chunking support for large date ranges
- **Training**: PyTorch Lightning with early stopping, checkpointing, and TensorBoard logging
- **Cloud Training**: Support for remote GPU training via Lightning AI cloud
- **Deployment**: Auto-upload best model to QuantConnect Object Store

## Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Alpaca API (required for data)
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# QuantConnect (required for model deployment)
QUANTCONNECT_USER_ID=your_user_id
QUANTCONNECT_API_TOKEN=your_token
QUANTCONNECT_ORG_ID=your_org_id  # optional

# Lightning AI (optional, for cloud training)
LIGHTNING_API_KEY=your_lightning_api_key
```

### 3. Login to Lightning AI (for Cloud Training)

If you want to use cloud GPU training:

```bash
lightning_app login
```

This will open your browser to authenticate with Lightning AI.

## Usage

### Cloud GPU Training (Default)

By default, `train.py` is configured to run on cloud/remote GPU servers:

```bash
# Default: runs on cloud GPU
python train.py
```

Or use the Lightning App for full cloud deployment:

```bash
# Using Lightning App (recommended for cloud)
lightning_app run app train_app.py
```

### Local Training (Mac with MPS)

To train on your local Mac using Apple's Metal Performance Shaders, use the `--local` flag:

```bash
python train.py --local
```

### Training Options

**Command-line arguments:**
- `--local`: Train on local Mac with MPS instead of cloud GPU (default: cloud GPU)

**Configuration (edit in `train.py`):**
All training parameters are configured at the top of `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DAYS` | 1460 | Days of history |
| `TIMEFRAME` | `TimeFrame.Minute` | `TimeFrame.Minute` or `TimeFrame.Hour` |
| `SEQ_LEN` | 60 | Lookback window |
| `PRED_LEN` | 1 | Prediction horizon |
| `D_MODEL` | 64 | Transformer hidden dimension |
| `NHEAD` | 4 | Number of attention heads |
| `NUM_LAYERS` | 2 | Number of transformer layers |
| `PATCH_LEN` | 8 | Length of each patch |
| `STRIDE` | 4 | Stride between patches |
| `BATCH_SIZE` | 32 | Batch size |
| `MAX_EPOCHS` | 50 | Max epochs |
| `LEARNING_RATE` | 1e-3 | Learning rate |
| `NUM_WORKERS` | 7 | Number of DataLoader worker processes |

### Predict

```bash
python predict.py --checkpoint checkpoints/patchtst-epoch=20-val_loss=0.0814.ckpt
```

Outputs metrics (MSE, RMSE, MAE, MAPE, Direction Accuracy) and plots predictions vs actuals.

## Project Structure

```
Training Gym V1/
├── Core Training Files
│   ├── train.py              # Main training script (local/remote GPU)
│   ├── train_app.py          # Lightning App wrapper for cloud training
│   ├── model.py              # PatchTST model definition
│   ├── dataset.py           # PyTorch Dataset and DataLoader creation
│   └── callbacks.py          # Custom callbacks (QuantConnect upload)
│
├── Data Processing
│   ├── data_loader.py        # Alpaca API client with chunking support
│   ├── download_data.py      # Standalone data download utility
│   └── feature_engineering.py # Feature engineering pipeline
│
├── Configuration & Documentation
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # This file
│   └── .env                  # Environment variables (create this)
│
├── Generated Directories
│   ├── checkpoints/          # Saved model weights (.ckpt files)
│   ├── logs/                 # TensorBoard logs
│   └── data/                 # Cached OHLCV data (CSV files)
│
└── venv/                     # Virtual environment (gitignored)
```

### File Descriptions

**Core Training:**
- `train.py`: Main training entry point. Supports both local (MPS) and cloud (GPU) training. Handles data loading, model creation, training loop, and evaluation.
- `train_app.py`: Lightning App wrapper that enables cloud GPU training on Lightning AI infrastructure. Automatically configures cloud compute resources.
- `model.py`: Contains `PatchTST` model and `TimeSeriesLightningModule` wrapper for PyTorch Lightning.
- `dataset.py`: Implements `TimeSeriesDataset` for sequence data and `create_dataloaders()` for train/val/test splits with normalization.
- `callbacks.py`: Custom PyTorch Lightning callback that automatically uploads the best model checkpoint to QuantConnect Object Store.

**Data Processing:**
- `data_loader.py`: Fetches crypto data from Alpaca API with automatic chunking for large date ranges. Handles multiple symbols and applies feature engineering.
- `download_data.py`: Standalone utility for downloading and caching historical data without running training.
- `feature_engineering.py`: Comprehensive feature engineering including time features, technical indicators (RSI, MACD, Bollinger Bands, etc.), and cross-symbol interconnectivity features.

## Feature Engineering

The `feature_engineering.py` module provides comprehensive feature engineering:

- **Time Features**: Cyclical encoding of hour, day of week, day of month, and month using sin/cos transformations
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and multiple moving averages (configurable periods)
- **Interconnectivity Features**: Cross-symbol features including:
  - Price ratios between symbol pairs (e.g., BTC/ETH ratio)
  - Relative returns (how each symbol performs vs others)
  - Rolling correlations between symbols
  - Spread features and relative strength

Features are automatically applied during training. The `engineer_features()` function can also be used standalone for live trading data preparation.

## Cloud Training with Lightning AI

This project supports training on Lightning AI cloud GPUs with minimal code changes:

1. **Install Lightning App**: Already included in `requirements.txt`
2. **Login**: Run `lightning_app login` to authenticate
3. **Run**: Use `lightning_app run app train_app.py` to train on cloud GPUs

The `train_app.py` wrapper automatically:
- Configures cloud GPU compute resources
- Uploads your code and data to Lightning AI
- Runs training on remote GPUs
- Handles infrastructure management

Benefits:
- No local GPU required
- Scalable compute resources
- Automatic infrastructure management
- Same code works locally and in cloud

## TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser to view:
- Training/validation loss curves
- Learning rate schedules
- Model hyperparameters
- Training metrics over time

## Model Deployment

After training, the best model checkpoint is automatically uploaded to QuantConnect Object Store via the `QuantConnectUploadCallback`. The model can then be loaded in QuantConnect algorithms for live trading.

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:

- **PyTorch & Lightning**: Deep learning framework and training utilities
- **Pandas & NumPy**: Data manipulation and numerical computing
- **scikit-learn**: Data preprocessing (StandardScaler)
- **Alpaca-py**: Crypto market data API
- **Lightning-app**: Cloud GPU training infrastructure
- **Matplotlib & TensorBoard**: Visualization and logging

## License

[Add your license here]
