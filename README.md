# Training Gym V1

A lightweight time-series forecasting toolkit for crypto price prediction. Train LSTM or Transformer models on hourly OHLCV data, then deploy to QuantConnect.

## Features

- **Models**: LSTM and Transformer architectures for time-series forecasting
- **Data**: Automatic download from Alpaca with local caching
- **Training**: PyTorch Lightning with early stopping, checkpointing, and TensorBoard logging
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
├── train.py         # Training script
├── predict.py       # Inference + visualization
├── model.py         # LSTM & Transformer models
├── dataset.py       # PyTorch dataset + dataloaders
├── data_loader.py   # Alpaca data fetching
├── callbacks.py     # QuantConnect upload callback
├── checkpoints/     # Saved model weights
├── logs/            # TensorBoard logs
└── data/            # Cached OHLCV data
```

## TensorBoard

```bash
tensorboard --logdir logs
```


