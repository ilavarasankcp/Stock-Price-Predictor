# Stock Price Predictor

Simple project to train a model that predicts next-day stock `Close` price from historical data.

Getting started

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train a model (example for Apple):

```bash
python train.py --ticker AAPL --start 2018-01-01 --model linear --out outputs
```

Outputs (saved to `outputs/`) include trained model and prediction plot.

Files

- `train.py`: CLI to download data, create features, train and evaluate models.
- `src/data_loader.py`: data download and feature engineering.
- `src/model.py`: model builders and save/load helpers.
- `requirements.txt`: Python dependencies.
