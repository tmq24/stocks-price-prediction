# DA-LLM: Dynamic Formulaic Alpha Generation with LLMs for Stock Prediction

Encoder-decoder Transformer trained on LLM-generated formulaic alphas that
regenerate dynamically during evaluation. Tickers AAPL / HSBC / PEP / TM / TCEHY,
70/30 chronological split.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` with `OPENAI_API_KEY` (required for alpha generation) and,
optionally, `EODHD_API_TOKEN` (only needed to re-download news).

## Usage

All commands go through the `thesis` subcommand of `main.py`.

```bash
# Full run: all 5 tickers, 70/30 split
python main.py thesis

# Single ticker (smoke test)
python main.py thesis --tickers AAPL

# Run an ablation (A1=drop C1, A2=drop C2, A3=drop C3, A5=paper static alphas)
python main.py thesis --ablation A1

# Generate alphas only, no training
python main.py thesis gen-alphas --tickers AAPL
```

Results are written under `results/`.
