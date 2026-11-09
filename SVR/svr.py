"""Portfolio adaptation of ENGI 9808 Lab 1: leakage-safe used-car price SVR.

Requires a locally supplied car_prices.csv (not redistributed in this repository).
Reported coursework results are historical and are NOT reproduced by this script.
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

CATEGORICAL = ["fuel type", "seller type", "transmission"]
TARGET = "selling price"


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if TARGET not in df or any(c not in df for c in CATEGORICAL):
        raise ValueError("CSV missing required columns")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    numerical = [c for c in X.columns if c not in CATEGORICAL]

    # Split first. No scaler or encoder sees the held-out test records.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numerical),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    # Small, transparent comparison—not the full 50-model coursework sweep.
    candidates = {
        "linear": {"kernel": "linear", "C": 1.0},
        "polynomial": {"kernel": "poly", "C": 1.0},
        "rbf": {"kernel": "rbf", "C": 5.0, "epsilon": 0.1, "tol": 1e-5, "max_iter": 10000},
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_model = None, float("inf"), None
    for name, settings in candidates.items():
        model = TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessing", preprocessing),
                ("svr", SVR(**settings)),
            ]),
            transformer=StandardScaler(),
        )
        fold_mse = -cross_val_score(
            model, X_dev, y_dev, cv=cv, scoring="neg_mean_squared_error"
        )
        rmse = float(np.sqrt(fold_mse).mean())
        print(f"{name}: mean 5-fold validation RMSE = {rmse:,.2f}")
        if rmse < best_score:
            best_name, best_score, best_model = name, rmse, model

    best_model.fit(X_dev, y_dev)
    predictions = best_model.predict(X_test)
    print(f"Selected kernel: {best_name}")
    print(f"Held-out test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(f"Held-out test R²: {r2_score(y_test, predictions):.4f}")
    print("Note: This redesigned evaluation is not the original lab's reported experiment.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("car_prices.csv"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    main(args.data)
