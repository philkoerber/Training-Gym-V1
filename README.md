# Training Gym V1

A lightweight time-series forecasting toolkit for crypto price prediction. Train LSTM or Transformer models on multi-symbol crypto data with advanced feature engineering, then deploy to QuantConnect.

## Features

- **Models**: LSTM and Transformer architectures for time-series forecasting
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

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
QUANTCONNECT_USER_ID=your_user_id
QUANTCONNECT_API_TOKEN=your_token
QUANTCONNECT_ORG_ID=your_org_id  # optional
LIGHTNING_API_KEY=your_lightning_api_key  # optional, for cloud training
```

## Usage

### Train

The training script automatically loads BTC/USD, ETH/USD, and LTC/USD data, applies feature engineering (time features, technical indicators, and interconnectivity features), and trains a model on the combined dataset.

**Local training (Mac with MPS):**
```bash
python train.py --days 60 --model lstm --max_epochs 50
```

**Cloud/Remote GPU training:**
```bash
python train.py --cloud --days 60 --model lstm --max_epochs 50
```

Options:
| Arg | Default | Description |
|-----|---------|-------------|
| `--days` | 1460 | Days of history |
| `--seq_len` | 60 | Lookback window |
| `--pred_len` | 1 | Prediction horizon |
| `--model` | lstm | `lstm` or `transformer` |
| `--hidden_dim` | 64 | Model hidden size |
| `--num_layers` | 2 | Number of layers |
| `--batch_size` | 32 | Batch size |
| `--max_epochs` | 50 | Max epochs |
| `--lr` | 1e-3 | Learning rate |
| `--timeframe` | minute | `minute` or `hour` |
| `--cloud` | False | Train on cloud/remote GPU (uses MPS locally if not set) |

### Predict

```bash
python predict.py --checkpoint checkpoints/lstm-epoch=20-val_loss=0.0814.ckpt
```

Outputs metrics (MSE, RMSE, MAE, MAPE, Direction Accuracy) and plots predictions vs actuals.

## Project Structure

```
├── train.py              # Training script
├── predict.py            # Inference + visualization
├── model.py              # LSTM & Transformer models
├── dataset.py            # PyTorch dataset + dataloaders
├── data_loader.py        # Alpaca data fetching with chunking support
├── feature_engineering.py # Feature engineering (time, technical, interconnectivity)
├── callbacks.py          # QuantConnect upload callback
├── checkpoints/          # Saved model weights
├── logs/                 # TensorBoard logs
└── data/                 # Cached OHLCV data
```

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

## TensorBoard

```bash
tensorboard --logdir logs
```


