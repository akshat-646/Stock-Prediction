# 🤖 Transformer Model — Nifty 50 Stock Prediction

> A multi-stock price forecasting system built on a **Transformer Encoder** architecture, trained across all 48 Nifty 50 tickers with few-shot and zero-shot generalization to unseen stocks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features & Technical Indicators](#features--technical-indicators)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference Modes](#inference-modes)
- [Future Forecasting](#future-forecasting)
- [Multi-Stock Batch Prediction](#multi-stock-batch-prediction)
- [Requirements](#requirements)
- [File Structure](#file-structure)

---

## Overview

This model predicts the **next 5 trading days' closing prices** for any Indian NSE-listed stock. It is trained on 15 years of historical OHLCV data for 48 Nifty 50 stocks and uses a Transformer Encoder with per-stock learnable embeddings to capture both temporal patterns and stock-specific characteristics.

**Key capabilities:**
- Multi-stock unified training (48 Nifty 50 tickers)
- Zero-shot inference for unseen stocks (using mean embedding)
- Few-shot adaptation for improved accuracy on unseen stocks (8 epochs)
- 5-day multi-step ahead forecasting
- Interactive Plotly visualizations

---

## Architecture

```
Input Features (13 features × 90 timesteps)
        ↓
Linear Projection  →  d_model = 64
        ↓
Positional Encoding (sinusoidal, max_len=500)
        ↓
Stock Embedding (48 stocks × 64 dims)  ← added per timestep
        ↓
Transformer Encoder Layer (d_model=64, nhead=4)  × 2 layers
        ↓
Last Token Output  [CLS-style]
        ↓
Fully Connected  →  OUTPUT_LEN = 5
        ↓
5-Day Predicted Close Prices (scaled)
```

### Key Components

| Component | Details |
|---|---|
| `PositionalEncoding` | Sinusoidal encoding, max length 500 |
| `TransformerEncoderLayer` | d_model=64, nhead=4, batch_first=True |
| `TransformerEncoder` | 2 stacked layers |
| `nn.Embedding` | One 64-dim vector per ticker (48 stocks) |
| Output | 5-step ahead close price prediction |

---

## Features & Technical Indicators

Each timestep uses **13 features** engineered from raw OHLCV data:

| Feature | Description |
|---|---|
| `open`, `high`, `low`, `close`, `volume` | Raw OHLCV |
| `return_1d` | 1-day percentage return |
| `return_5d` | 5-day percentage return |
| `sma_20` | 20-day Simple Moving Average |
| `ema_12` | 12-day Exponential Moving Average |
| `macd` | EMA12 − EMA26 |
| `rsi` | 14-period Relative Strength Index |
| `bb_width` | Bollinger Band width (2σ / SMA20) |
| `volatility` | Rolling 20-day return std deviation |
| `volume_ratio` | Volume / 20-day average volume |

All features are normalized per-ticker using **MinMaxScaler**.

---

## Configuration

```python
INPUT_LEN   = 90        # Lookback window (90 trading days ≈ 4.5 months)
OUTPUT_LEN  = 5         # Forecast horizon (5 trading days ≈ 1 week)
EPOCHS      = 25        # Training epochs
BATCH_SIZE  = 64        # Training batch size
d_model     = 64        # Transformer hidden dimension
nhead       = 4         # Attention heads
num_layers  = 2         # Transformer encoder layers
```

---

## Data Pipeline

### `DataPipeline` Class

```
yf.download(ticker, period="15y")
        ↓
Rename & select [open, high, low, close, volume]
        ↓
add_features() → 13-feature DataFrame
        ↓
MinMaxScaler (fit on train split)
        ↓
create_sequences(INPUT_LEN=90, OUTPUT_LEN=5)
        ↓
Train / Test split (80% / 20%)
```

- Data source: **Yahoo Finance** via `yfinance`
- Historical period: **15 years**
- Train/Test split: **80% / 20%**
- Sequences: sliding window of 90 days → predict next 5 days

### `prepare_single(ticker)` Method
Used for unseen (out-of-vocabulary) stocks. Fits a fresh scaler on the ticker's own data and returns train/test sequences and a DatetimeIndex for forecasting.

---

## Training

```python
optimizer = Adam(lr=0.0003)
loss      = MSELoss()
scheduler = ReduceLROnPlateau(patience=3, factor=0.5)
grad_clip = 1.0
```

Training pipeline:
1. Fetches 15-year OHLCV data for all 48 NIFTY 50 tickers
2. Engineers 13 technical features per ticker
3. Creates sliding-window sequences (INPUT_LEN=90, OUTPUT_LEN=5)
4. Trains unified model with per-stock embeddings
5. Saves checkpoint to `nifty50_model.pt`

**Checkpoint contents:**
```python
{
    "model_state": model.state_dict(),
    "input_size":  INPUT_FEATS,
    "scalers":     pipeline.scalers,    # dict of MinMaxScalers
    "date_map":    pipeline.date_map,   # dict of DatetimeIndex
}
```

> ⚡ To skip training, upload `nifty50_model.pt` and the notebook will load it automatically.

---

## Inference Modes

### 1. Zero-Shot Inference
For completely unseen stocks, uses the **mean of all learned stock embeddings** as a proxy:

```python
mean_emb = model.embed.weight.mean(dim=0, keepdim=True)
preds = run_inference(model, X_test, emb_override=mean_emb)
```

### 2. Few-Shot Adaptation
Trains a **new embedding vector** for the unseen stock while keeping all model weights frozen:

```python
# Freeze all model weights
for param in fs_model.parameters():
    param.requires_grad = False

# Only train the new embedding
unseen_emb = nn.Parameter(mean_emb.clone(), requires_grad=True)
optimizer  = Adam([unseen_emb], lr=0.001)
# Train for 8 epochs on unseen stock's train split
```

This adapts the stock-specific bias in ~8 epochs without retraining the whole model.

---

## Future Forecasting

After few-shot adaptation, generates a **5-day forward forecast**:

```python
# Use the last 90-day window from test data
last_seq = X_unseen_test[-1]  # shape: (90, 13)

# Forward pass with adapted embedding
fut_scaled = fs_model(last_seq, unseen_emb)   # 5 scaled values
fut_prices = inverse_close(fut_scaled, ticker) # back to ₹
```

Output format:
```
Last known close: ₹1,234.56  (date: 2025-01-10)

  2025-01-13  →  ₹1,241.20
  2025-01-14  →  ₹1,256.80
  2025-01-15  →  ₹1,249.30
  2025-01-16  →  ₹1,263.10
  2025-01-17  →  ₹1,271.50
```

An interactive **Plotly chart** shows the last 30 days of actual vs. predicted prices connected to the 5-day forecast.
For Example:
<img width="1266" height="655" alt="image" src="https://github.com/user-attachments/assets/6cb3b856-8e65-4e44-af9b-fc2e72920c05" />

---

## Multi-Stock Batch Prediction

The notebook supports batch processing of multiple unseen tickers:

```python
BATCH_UNSEEN = ["PIDILITIND.NS", "HAVELLS.NS", "DABUR.NS"]
```

For each ticker the pipeline:
1. Fetches and prepares data via `prepare_single()`
2. Runs zero-shot inference with `mean_emb`
3. Computes metrics (MAPE, R²)
4. Generates 5-day forward forecast
5. Prints a summary metrics table

---

## Requirements

```
yfinance
plotly
scikit-learn
torch
numpy
pandas
```

Install:
```bash
pip install yfinance plotly scikit-learn torch
```

---

## File Structure

```
TRANSFORM_MODEL_STOCK_PREDICTION.ipynb   ← Main notebook
nifty50_model.pt                         ← Saved model checkpoint (generated after training)
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MAPE** | Mean Absolute Percentage Error (lower is better) |
| **R²** | Coefficient of Determination (closer to 1.0 is better) |
| **MAE** | Mean Absolute Error in ₹ |
| **RMSE** | Root Mean Squared Error in ₹ |

---

## Notes

- The model is trained exclusively on **NSE (National Stock Exchange)** tickers using the `.NS` suffix (e.g., `RELIANCE.NS`).
- For BSE tickers, use the `.BO` suffix.
- Predictions are **not financial advice**. This is a research/educational model.
