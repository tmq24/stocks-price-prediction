# Stock Close Price Prediction Benchmark

This repository implements a benchmark for predicting stock **Log Close Prices** using various Deep Learning architectures. The project uses 23 technical indicators as features and evaluates models on a per-ticker basis.

## Overview

- **Task**: Predict Log Close Price at horizon $t+h$.
- **Features**: 23 technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.).
- **Data Split**:
  - **Train**: 2015-2021 (Combined across tickers).
  - **Validation**: 2022-2023 (Combined across tickers).
  - **Test**: 2024-2025 (Evaluated per-ticker: AAPL, HSBC, PEP, TM, TCEHY).

## Models Implemented

1. **LSTM**: Standard Long Short-Term Memory network.
2. **N-BEATS**: Neural basis expansion analysis for interpretable time series forecasting (Trend, Seasonality, and Generic blocks).
3. **Transformer Encoder-Only**: Uses the encoder stack and the last token for prediction.
4. **Transformer Decoder-Only**: GPT-style causal transformer.
5. **Vanilla Transformer**: Full Encoder-Decoder architecture.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stocks-trend-prediction.git
cd stocks-trend-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project uses a CLI defined in `main.py`.

### Run Full Benchmark

Train and evaluate all models across all windows (5, 10, 15) and horizons (1, 5, 10):

```bash
python main.py benchmark
```

- Results are saved to `evaluation/benchmark_results.csv`.
- Plots for `AAPL` and `PEP` are saved to `plot/`.

### Run Single Experiment

Train a specific model with custom parameters:

```bash
python main.py single --model lstm --window 10 --horizon 1
python main.py single --model transformer_encoder --window 10 --horizon 1
python main.py single --model transformer_decoder --window 10 --horizon 1
python main.py single --model vanilla_transformer --window 10 --horizon 1
python main.py single --model nbeats --window 10 --horizon 1
```

### Quick Test

Run a fast test with only 2 models:

```bash
python main.py test
```

## Evaluation Metrics

- **MAE**: Mean Absolute Error (on scaled log prices).
- **MSE**: Mean Squared Error (on scaled log prices).
- **Correlation**: Pearson correlation between predicted and actual movements.
- **Plots**: Real price ($) comparison for visual verification.
