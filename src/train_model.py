import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .feature_engineering import add_time_features, build_preprocessor
from .utils import save_model, save_metadata, compute_metrics, ensure_dir

import json
import warnings
warnings.filterwarnings("ignore")


def train_and_select_model(processed_csv: str = None) -> dict:
    """Train Linear Regression, Random Forest, and (if available) XGBoost models.
    Returns a dictionary containing the best model, its name, and performance metrics.
    """
    # Load data
    if processed_csv is None:
        base_dir = Path(__file__).resolve().parents[2]
        processed_csv = base_dir / "data" / "processed_data.csv"
    df = pd.read_csv(processed_csv, parse_dates=["date"])

    # Feature engineering
    df = add_time_features(df)
    X = df.drop(columns=["tourist_count", "date"])
    y = df["tourist_count"]

    # Build preprocessing pipeline (one‑hot for destination, pass‑through numeric)
    preprocessor = build_preprocessor(df)
    # We'll fit the preprocessor inside each model pipeline

    # Train‑test split (80/20 random, acceptable for regression task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # Helper to fit a pipeline and evaluate
    def fit_evaluate(name, estimator):
        from sklearn.pipeline import Pipeline
        pipe = Pipeline(steps=[("preproc", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = compute_metrics(y_test.values, preds)
        results.append({"name": name, "pipeline": pipe, "metrics": metrics})

    # Linear Regression
    fit_evaluate("Linear Regression", LinearRegression())

    # Random Forest
    fit_evaluate("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))

    # XGBoost if available
    try:
        from xgboost import XGBRegressor
        fit_evaluate("XGBoost", XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
    except Exception as e:
        # XGBoost not installed – just skip
        pass

    # Choose best model based on RMSE (lower is better)
    best = min(results, key=lambda r: r["metrics"]["RMSE"])

    # Save model and metadata
    model_dir = Path(__file__).resolve().parents[2] / "models"
    ensure_dir(str(model_dir))
    model_path = model_dir / "tourism_model.pkl"
    metadata_path = model_dir / "model_metadata.json"
    save_model(best["pipeline"], str(model_path))
    # Include metrics for all models for reference
    all_metrics = {r["name"]: r["metrics"] for r in results}
    metadata = {
        "best_model": best["name"],
        "metrics": all_metrics,
        "training_date": pd.Timestamp.now().isoformat(),
    }
    save_metadata(metadata, str(metadata_path))
    return {
        "best_model_name": best["name"],
        "best_pipeline": best["pipeline"],
        "metrics": best["metrics"],
    }

if __name__ == "__main__":
    res = train_and_select_model()
    print(json.dumps(res["metrics"], indent=2))
