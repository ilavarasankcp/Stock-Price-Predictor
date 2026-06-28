from typing import Tuple
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_model() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    return pipe


def build_random_forest(n_estimators: int = 100) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=42))
    ])
    return pipe


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
