# 📊 Nifty 50 Stock Prediction — Overall Process & Model Comparison

> A comprehensive guide covering the end-to-end pipeline shared by both models, and a detailed comparison between the **Transformer Encoder** and **LSTM** approaches for multi-stock price forecasting.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Shared Pipeline Architecture](#shared-pipeline-architecture)
- [Step-by-Step Process](#step-by-step-process)
- [Model Comparison](#model-comparison)
- [Architecture Diagrams](#architecture-diagrams)
- [When to Use Which Model](#when-to-use-which-model)
- [Strengths & Limitations](#strengths--limitations)
- [Quick Start Guide](#quick-start-guide)
- [Glossary](#glossary)

---

## Project Overview

Both models solve the same problem using different neural architectures:

> **Given the last 90 trading days of OHLCV + technical indicators for a stock, predict the next 5 closing prices.**

Both notebooks share:
- The same **48 Nifty 50 training tickers**
- The same **13-feature engineering pipeline**
- The same **few-shot / zero-shot adaptation strategy**
- The same **evaluation metrics** (MAPE, R²)
- The same **Plotly visualization** output

They differ in: **how they process the sequential input** (attention vs. recurrence).

---

## Shared Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SHARED DATA PIPELINE                    │
│                                                             │
│  Yahoo Finance API  →  OHLCV (15y / 10y)                    │
│         ↓                                                   │
│  Feature Engineering  →  13 features per timestep           │
│         ↓                                                   │
│  MinMaxScaler  (per-ticker, fit on train split)             │
│         ↓                                                   │
│  Sliding Window Sequences  →  (90 days IN, 5 days OUT)      │
│         ↓                                                   │
│  80/20 Train-Test Split                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │                                     │
   TRANSFORMER MODEL                    LSTM MODEL
   nifty50_model.pt                     nifty50_lstm_model.pt
        │                                     │
        └──────────────┬──────────────────────┘
                       ↓
        ┌─────────────────────────────────────┐
        │       SHARED INFERENCE PIPELINE     │
        │                                     │
        │  1. Zero-Shot (mean embedding)      │
        │  2. Few-Shot Adaptation (8 epochs)  │
        │  3. 5-Day Future Forecast           │
        │  4. Inverse Scale → ₹ prices        │
        │  5. Plotly Chart                    │
        └─────────────────────────────────────┘
```

---

## Step-by-Step Process

### Step 1 — Data Acquisition

```python
df = yf.download(ticker, period="15y")   # Transformer
df = yf.download(ticker, period="10y")   # LSTM
```

Raw columns: `open`, `high`, `low`, `close`, `volume`

### Step 2 — Feature Engineering

13 features are computed for each trading day:

```
OHLCV (5)  +  Returns (2)  +  Moving Averages (3)  +  Oscillators (3)
```

| Category | Features |
|---|---|
| Raw price | open, high, low, close, volume |
| Returns | return_1d, return_5d |
| Trend | sma_20, ema_12, macd |
| Momentum | rsi, bb_width |
| Risk | volatility, volume_ratio |

### Step 3 — Normalization

Each ticker's features are independently scaled to [0, 1] using `MinMaxScaler`. The scaler is fit **only on the training split** to prevent data leakage.

### Step 4 — Sequence Creation

A sliding window of 90 days produces one training sample:

```
[day_1, day_2, ..., day_90]  →  [day_91, day_92, day_93, day_94, day_95]
         INPUT (X)                           TARGET (y)
    shape: (90, 13)                        shape: (5,)  ← close prices only
```

### Step 5 — Unified Multi-Stock Training

Both models are trained on **all 48 tickers simultaneously** using a stock ID embedding that allows the model to learn shared patterns while maintaining per-stock identity.

```python
# Each training sample includes:
X  → (batch, 90, 13)   # sequence features
y  → (batch, 5)         # target close prices
sid → (batch,)          # stock ID (0–47)
```

### Step 6 — Model Training

```
Optimizer:   Adam (lr=0.0003)
Loss:        MSELoss
Clip Grad:   1.0
Epochs:      25
Batch Size:  64
Scheduler:   ReduceLROnPlateau (patience=3, factor=0.5)
```

### Step 7 — Zero-Shot Inference (Unseen Stocks)

When a new ticker not seen during training is provided:
- Compute mean of all 48 learned stock embeddings
- Use this as a proxy embedding for the new stock
- No retraining required — runs inference directly

### Step 8 — Few-Shot Adaptation

For improved accuracy on unseen tickers:
- **Freeze** all model weights
- Initialize a new embedding vector from the mean embedding
- **Only train the new embedding** for 8 epochs on the unseen stock's training split

```python
unseen_emb = nn.Parameter(mean_emb.clone(), requires_grad=True)
# All other parameters: requires_grad = False
optimizer  = Adam([unseen_emb], lr=0.001)
```

### Step 9 — 5-Day Forecast

Take the last 90-day window from the unseen stock, pass through the adapted model, and inverse-transform the 5 predicted scaled values back to ₹ prices.

### Step 10 — Visualization

Interactive Plotly chart showing:
- Last 30 days: Actual prices (cyan)
- Last 30 days: Few-shot predicted prices (orange)
- Next 5 days: Future forecast (green dashed)

---

## Model Comparison

### Side-by-Side Architecture

| Aspect | Transformer Model | LSTM Model |
|---|---|---|
| **Core mechanism** | Self-attention (all timesteps in parallel) | Recurrence (sequential timestep processing) |
| **Hidden dimension** | 64 (d_model) | 128 (hidden_size) |
| **Depth** | 2 Transformer Encoder layers | 2 LSTM layers |
| **Attention heads** | 4 (multi-head attention) | N/A |
| **Positional info** | Sinusoidal Positional Encoding | Implicit via hidden state carry-over |
| **Stock embedding size** | 64 | 128 |
| **Dropout** | Default (in TransformerEncoderLayer) | 0.2 (between LSTM layers) |
| **Parameter count** | ~Lower (smaller hidden dim) | ~Higher (larger hidden dim) |
| **Training data period** | 15 years | 10 years |
| **Checkpoint file** | `nifty50_model.pt` | `nifty50_lstm_model.pt` |

---

### Conceptual Comparison

| Property | Transformer | LSTM |
|---|---|---|
| **Sequence modeling** | Global — attends to all 90 days simultaneously | Local — processes day by day, gating what to remember |
| **Long-range dependencies** | Excellent — direct attention between any two days | Good — but can suffer from vanishing gradient over 90 steps |
| **Training speed** | Faster — parallelizable over sequence length | Slower — sequential computation per timestep |
| **Interpretability** | Attention maps can show which days influenced the prediction | Hidden state is opaque |
| **Data efficiency** | Needs more data to learn attention patterns well | Can work with less data |
| **Few-shot adaptation** | New embedding adapts the 64-dim projection space | New embedding adapts the 128-dim LSTM input space |
| **Overfitting risk** | Lower (attention regularizes through multi-head averaging) | Moderate (mitigated by dropout=0.2) |
| **Inductive bias** | Minimal — learns temporal structure from attention | Strong — built-in recurrence is biased toward sequential order |

---

### Training Data Difference

| Model | Historical Period | Approximate Samples per Ticker |
|---|---|---|
| Transformer | 15 years | ~3,600 trading days |
| LSTM | 10 years | ~2,400 trading days |

The Transformer uses 5 more years of data, which helps its data-hungry attention mechanism learn more robust long-range patterns.

---

### Few-Shot Adaptation Comparison

Both models use the **identical few-shot strategy**, but the adaptation lives in different spaces:

```
Transformer Few-Shot:
  unseen_emb shape: (1, 64)   → added to projected 64-dim input before attention

LSTM Few-Shot:
  unseen_emb shape: (1, 128)  → added to projected 128-dim input before recurrence
```

The optimization loop is identical — only the embedding dimensionality and the downstream computation graph differ.

---

### Summary Table

| Criterion | Transformer | LSTM | Winner |
|---|---|---|---|
| Parallelism during training | ✅ High | ❌ Sequential | Transformer |
| Long-range dependency capture | ✅ Excellent | ✅ Good | Transformer |
| Works well with less data | ❌ Needs more data | ✅ Better | LSTM |
| Interpretability (attention) | ✅ Yes | ❌ No | Transformer |
| Parameter efficiency | ✅ Smaller model | ❌ Larger model | Transformer |
| Historical data used | 15 years | 10 years | Transformer |
| Proven track record in time-series | ✅ Emerging | ✅ Established | Tie |
| Few-shot adaptability | ✅ Both identical strategy | ✅ Both identical strategy | Tie |

---

## Architecture Diagrams

### Transformer Flow

```
Input (B, 90, 13)
    → Linear Proj     (B, 90, 64)
    + Positional Enc  (90, 64)
    + Stock Embed     (64)         ← broadcast over T
    → TransformerEnc  (B, 90, 64)  [2 layers, 4 heads]
    → Last Token      (B, 64)
    → FC              (B, 5)
```

### LSTM Flow

```
Input (B, 90, 13)
    → Linear Proj     (B, 90, 128)
    + Stock Embed     (128)         ← broadcast over T
    → LSTM            (B, 90, 128)  [2 layers, dropout=0.2]
    → Last h_T        (B, 128)
    → FC              (B, 5)
```

---

## When to Use Which Model

**Use the Transformer model when:**
- You have access to 15+ years of training data
- Training speed and parallelism matter
- You want to inspect attention maps for model explainability
- You're experimenting with multi-head attention for financial time-series
- Computational resources (GPU) are limited (smaller model)

**Use the LSTM model when:**
- You have limited historical data (10 years is sufficient)
- You prefer a well-established, battle-tested time-series architecture
- You want a larger hidden representation (128 vs 64) for complex patterns
- Sequential dependencies within a trading session are the primary signal

**Both models are equivalent for:**
- Zero-shot prediction on unseen tickers
- Few-shot adaptation to new stocks
- Multi-step (5-day) forecasting
- Nifty 50 multi-ticker unified training

---

## Strengths & Limitations

### Strengths (Both Models)

- **Multi-stock generalization**: One model serves 48+ tickers
- **Few-shot adaptation**: New stocks require only 8 epochs of lightweight fine-tuning
- **Rich feature set**: 13 technical indicators capture trend, momentum, and volatility
- **End-to-end pipeline**: From raw Yahoo Finance data to ₹ forecast in a single notebook
- **Checkpoint support**: Skip retraining by uploading a saved `.pt` file

### Limitations (Both Models)

- **No fundamental data**: Earnings, P/E ratio, news sentiment are not included
- **No macroeconomic inputs**: Interest rates, FII/DII flows, global indices not used
- **Market regime shifts**: Both models may underperform during black swan events (pandemics, geopolitical shocks)
- **Fixed horizon**: 5-day ahead only; extending to 10 or 20 days requires retraining
- **NSE-centric**: Optimized for Indian equity markets; may need re-tuning for other exchanges

---

## Quick Start Guide

### 1. Clone / Open the notebooks

- `TRANSFORM_MODEL_STOCK_PREDICTION.ipynb` — Transformer
- `LSTM_Stock_Prediction.ipynb` — LSTM

### 2. Install dependencies

```bash
pip install yfinance plotly scikit-learn torch
```

### 3. (Optional) Upload saved checkpoints

| Model | Checkpoint file |
|---|---|
| Transformer | `nifty50_model.pt` |
| LSTM | `nifty50_lstm_model.pt` |

If no checkpoint is found, training starts from scratch (~25 epochs × 48 stocks).

### 4. Run all cells

Both notebooks auto-detect GPU (`cuda`) and fall back to CPU.

### 5. Enter an unseen ticker when prompted

```
Stock Name: PIDILITIND.NS
```

The notebook will:
- Fetch data
- Run few-shot adaptation
- Print 5-day forecast with ₹ prices
- Render an interactive Plotly chart

---

## Glossary

| Term | Definition |
|---|---|
| **OHLCV** | Open, High, Low, Close, Volume — standard price data format |
| **Lookback window** | The 90-day input sequence used to make predictions |
| **Embedding** | A learnable vector that encodes stock identity |
| **Few-shot** | Adapting to a new stock using a small amount of its own data |
| **Zero-shot** | Making predictions for a new stock with no adaptation |
| **MAPE** | Mean Absolute Percentage Error — average % error across predictions |
| **R²** | R-squared — proportion of variance in targets explained by the model |
| **MinMaxScaler** | Normalizes features to the [0, 1] range per ticker |
| **Sliding window** | A technique to create sequences by moving a fixed-size window along the time axis |
| **d_model** | The hidden dimension of the Transformer (64 in this project) |
| **hidden_size** | The hidden dimension of the LSTM (128 in this project) |
