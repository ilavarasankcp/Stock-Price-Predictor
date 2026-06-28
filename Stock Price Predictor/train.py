import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import download_stock_data, create_features
from src.model import build_linear_model, build_random_forest, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train(ticker: str, start: str, end: str, model_type: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading data for {ticker}...")
    df = download_stock_data(ticker, start=start, end=end)
    data = create_features(df)

    features = [c for c in data.columns if c not in ("Target",)]
    X = data[features].values
    y = data["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == "linear":
        model = build_linear_model()
    else:
        model = build_random_forest()

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # compute RMSE in a way compatible with older and newer scikit-learn
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")

    save_path = os.path.join(out_dir, f"{ticker}_{model_type}_model.joblib")
    save_model(model, save_path)
    print(f"Saved model to {save_path}")

    # Plot predictions vs actual
    dates = data.index[-len(y_test):]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_test, label="Actual")
    plt.plot(dates, preds, label="Predicted")
    plt.legend()
    plt.title(f"{ticker} - Actual vs Predicted")
    fig_path = os.path.join(out_dir, f"{ticker}_{model_type}_pred.png")
    plt.savefig(fig_path)
    print(f"Saved plot to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Train stock price predictor")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model", choices=["linear", "rf"], default="linear")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    # pass selected model string directly to train
    train(args.ticker, args.start, args.end, args.model, args.out)


if __name__ == "__main__":
    main()
