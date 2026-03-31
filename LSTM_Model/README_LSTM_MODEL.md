# 📈 LSTM Model — Nifty 50 Stock Prediction

> A multi-stock price forecasting system built on a **Long Short-Term Memory (LSTM)** network, trained across all 48 Nifty 50 tickers with few-shot and zero-shot generalization to unseen stocks.

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

This model predicts the **next 5 trading days' closing prices** for any Indian NSE-listed stock. It is trained on 10 years of historical OHLCV data for 48 Nifty 50 stocks using a 2-layer LSTM with per-stock learnable embeddings. The LSTM captures sequential temporal dependencies in stock price movements, making it well-suited for time-series forecasting.

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
Linear Projection  →  hidden_size = 128
        ↓
Stock Embedding (48 stocks × 128 dims)  ← added per timestep
        ↓
LSTM  (hidden_size=128, num_layers=2, dropout=0.2)
        ↓
Last Hidden State  h_T
        ↓
Fully Connected  →  OUTPUT_LEN = 5
        ↓
5-Day Predicted Close Prices (scaled)
```

### Key Components

| Component | Details |
|---|---|
| `nn.Linear` (proj) | Projects input features → hidden_size (128) |
| `nn.Embedding` | One 128-dim vector per ticker (48 stocks) |
| `nn.LSTM` | hidden_size=128, num_layers=2, dropout=0.2 |
| Output FC | hidden_size → OUTPUT_LEN (5) |

### Architecture Design Rationale

- **Linear Projection**: Maps raw 13-feature input to the LSTM's hidden space, enabling the embedding and LSTM to operate in the same dimensionality.
- **Stock Embedding**: Added to every timestep of the projected input — allows the model to learn per-stock biases while sharing all LSTM weights across tickers.
- **2-Layer LSTM**: Captures both low-level price dynamics (layer 1) and higher-level trend patterns (layer 2).
- **Dropout (0.2)**: Applied between LSTM layers to prevent overfitting.

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
INPUT_LEN   = 90         # Lookback window (90 trading days ≈ 4.5 months)
OUTPUT_LEN  = 5          # Forecast horizon (5 trading days ≈ 1 week)
EPOCHS      = 25         # Training epochs
BATCH_SIZE  = 64         # Training batch size
HIDDEN_SIZE = 128        # LSTM hidden dimension
NUM_LAYERS  = 2          # LSTM stacked layers
```

---

## Data Pipeline

### `DataPipeline` Class

```
yf.download(ticker, period="10y")
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
- Historical period: **10 years**
- Train/Test split: **80% / 20%**
- Sequences: sliding window of 90 days → predict next 5 days

### `prepare_single(ticker)` Method
Used for unseen (out-of-vocabulary) stocks. Fits a fresh `MinMaxScaler` on the new ticker's own training split and returns sequences and a DatetimeIndex for forecasting.

---

## Training

```python
optimizer = Adam(lr=0.0003)
loss      = MSELoss()
scheduler = ReduceLROnPlateau(patience=3, factor=0.5)
grad_clip = 1.0
```

Training pipeline:
1. Fetches 10-year OHLCV data for all 48 Nifty 50 tickers
2. Engineers 13 technical features per ticker
3. Creates sliding-window sequences (INPUT_LEN=90, OUTPUT_LEN=5)
4. Trains unified LSTM model with per-stock embeddings
5. Saves checkpoint to `nifty50_lstm_model.pt`

**Checkpoint contents:**
```python
{
    "model_state": model.state_dict(),
    "input_size":  INPUT_FEATS,
    "scalers":     pipeline.scalers,    # dict of MinMaxScalers
    "date_map":    pipeline.date_map,   # dict of DatetimeIndex
}
```

> ⚡ To skip training, upload `nifty50_lstm_model.pt` and the notebook will load it automatically.

---

## Inference Modes

### 1. Zero-Shot Inference
For completely unseen stocks, uses the **mean of all learned stock embeddings** as a proxy:

```python
mean_emb = model.embed.weight.mean(dim=0, keepdim=True)  # (1, 128)

# Manual forward pass with embedding override:
xp       = model.proj(x)                            # (1, T, 128)
xp       = xp + mean_emb.unsqueeze(1)               # broadcast over time
out, _   = model.lstm(xp)                           # (1, T, 128)
pred     = model.fc(out[:, -1, :])                  # (1, 5)
```

### 2. Few-Shot Adaptation
Trains a **new embedding vector** for the unseen stock while keeping all LSTM weights frozen:

```python
# Freeze all model weights
for param in fs_model.parameters():
    param.requires_grad = False

# Only train the new embedding
unseen_emb = nn.Parameter(mean_emb.clone(), requires_grad=True)
optimizer  = Adam([unseen_emb], lr=0.001)

# Forward pass during few-shot training:
xp       = fs_model.proj(xb)
xp       = xp + unseen_emb.unsqueeze(1).expand(batch, -1, -1)
out, _   = fs_model.lstm(xp)
pred     = fs_model.fc(out[:, -1, :])
```

This adapts the stock-specific bias in ~8 epochs without modifying any LSTM weights.

---

## Future Forecasting

After few-shot adaptation, generates a **5-day forward forecast**:

```python
# Use the last 90-day window from test data
last_seq = X_unseen_test[-1]  # shape: (90, 13)

# Forward pass with adapted embedding
xp         = fs_model.proj(last_seq)
xp         = xp + unseen_emb.detach().unsqueeze(1)
out, _     = fs_model.lstm(xp)
fut_scaled = fs_model.fc(out[:, -1, :])          # 5 scaled values
fut_prices = inverse_close(fut_scaled, ticker)    # back to ₹
```

Output format:
```
Last known close: ₹1,234.56  (date: 2025-01-10)

  2025-01-13  →  ₹1,240.10  (+0.45%)
  2025-01-14  →  ₹1,255.30  (+1.68%)
  2025-01-15  →  ₹1,248.70  (+1.15%)
  2025-01-16  →  ₹1,261.90  (+2.21%)
  2025-01-17  →  ₹1,270.40  (+2.90%)
```

An interactive **Plotly chart** shows the last 30 days of actual vs. few-shot predicted prices connected to the 5-day forecast.
For Example:
<img width="1272" height="629" alt="image" src="https://github.com/user-attachments/assets/93c929ee-08f4-432e-b7f7-5959fc2df2e4" />

---

## Multi-Stock Batch Prediction

The notebook supports processing an arbitrary list of unseen tickers:

```python
UNSEEN_TICKERS_LIST = ["DMART.NS", "DABUR.NS", "INFY.BO"]
```

For each ticker the pipeline:
1. Fetches and prepares data via `prepare_single()`
2. Runs few-shot adaptation (8 epochs, lr=0.001)
3. Runs inference and computes metrics (MAPE, R²)
4. Generates 5-day forward forecast with change %
5. Generates an interactive Plotly chart per stock

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
LSTM_Stock_Prediction.ipynb         ← Main notebook
nifty50_lstm_model.pt               ← Saved model checkpoint (generated after training)
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

- The model uses 10 years of training data vs. the Transformer's 15 years — shorter history, but sufficient for the LSTM's sequential learning paradigm.
- The model is trained on **NSE tickers** (`.NS` suffix). BSE tickers use `.BO`.
- The LSTM hidden size (128) is larger than the Transformer d_model (64) to compensate for the lack of self-attention.
- Predictions are **not financial advice**. This is a research/educational model.
