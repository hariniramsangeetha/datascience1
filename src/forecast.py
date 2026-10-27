import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .feature_engineering import add_time_features, build_preprocessor
from .utils import save_model, save_metadata, compute_metrics, ensure_dir


def create_lag_features(df: pd.DataFrame, lags: list[int] = [1, 12]) -> pd.DataFrame:
    """Add lagged ``tourist_count`` columns.
    ``lags`` are in months; the data must be sorted by ``date``.
    """
    df = df.copy().sort_values("date")
    for lag in lags:
        df[f"tourist_lag_{lag}"] = df["tourist_count"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


def train_forecast_model(df: pd.DataFrame, destination: str) -> Pipeline:
    """Train a time‑series forecasting model for a specific destination.
    Uses a GradientBoostingRegressor (fallback to RandomForest if needed).
    Returns a fitted sklearn ``Pipeline`` that includes preprocessing.
    """
    # Filter destination and add features
    sub = df[df["destination"] == destination].reset_index(drop=True)
    sub = add_time_features(sub)
    sub = create_lag_features(sub)
    X = sub.drop(columns=["tourist_count", "date"])
    y = sub["tourist_count"]
    # Preprocessor (handle categorical destination – now constant, but keep for safety)
    preprocessor = build_preprocessor(sub)
    # Use a simple time‑series split (e.g., 5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    best_pipe = None
    best_rmse = float("inf")
    for model in [
        GradientBoostingRegressor(random_state=42),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    ]:
        pipe = Pipeline([("preproc", preprocessor), ("model", model)])
        rmses = []
        for train_idx, test_idx in tscv.split(X):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipe.predict(X.iloc[test_idx])
            rmse = np.sqrt(((preds - y.iloc[test_idx]) ** 2).mean())
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_pipe = pipe
    # Fit on full data
    best_pipe.fit(X, y)
    # Save model for future use
    model_dir = Path(__file__).resolve().parents[2] / "models"
    ensure_dir(str(model_dir))
    model_path = model_dir / f"forecast_{destination.lower().replace(' ', '_')}.pkl"
    save_model(best_pipe, str(model_path))
    # Save simple metadata
    meta_path = model_dir / f"forecast_{destination.lower().replace(' ', '_')}_meta.json"
    save_metadata({"destination": destination, "rmse": best_rmse}, str(meta_path))
    return best_pipe


def forecast_future(df: pd.DataFrame, destination: str, periods: int = 6) -> pd.DataFrame:
    """Generate a forecast for the next *periods* months.
    Returns a DataFrame with ``month`` and ``predicted_tourists`` columns.
    """
    # Train (or load) model for the destination
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_path = model_dir / f"forecast_{destination.lower().replace(' ', '_')}.pkl"
    if model_path.exists():
        from .utils import load_model
        pipe = load_model(str(model_path))
    else:
        pipe = train_forecast_model(df, destination)

    # Prepare the latest row for recursive forecasting
    sub = df[df["destination"] == destination].sort_values("date").reset_index(drop=True)
    sub = add_time_features(sub)
    sub = create_lag_features(sub)
    last_row = sub.iloc[-1].copy()
    forecasts = []
    current_date = pd.to_datetime(last_row["date"]).replace(day=1)
    for i in range(1, periods + 1):
        # Advance month
        current_date = current_date + pd.DateOffset(months=1)
        # Build a feature dict for the next month
        feature = {
            "destination": destination,
            "year": current_date.year,
            "month": current_date.month,
            "temperature": last_row["temperature"],  # keep last known weather as placeholder
            "rainfall": last_row["rainfall"],
            "humidity": last_row["humidity"],
            "holiday": int(current_date.month in {12, 1, 5, 8}),
            "festival": int(current_date.month in {3, 10, 11}),
            "weekend": 0,  # not used in model
            "hotel_price": last_row["hotel_price"],
            "hotel_occupancy": last_row["hotel_occupancy"],
            "transport_availability": last_row["transport_availability"],
            "previous_month_tourists": last_row["tourist_count"],
        }
        # Add time sinusoid features
        feature["month_sin"] = np.sin(2 * np.pi * feature["month"] / 12)
        feature["month_cos"] = np.cos(2 * np.pi * feature["month"] / 12)
        # Compute lag features for the model
        if i == 1:
            recent_counts = list(sub["tourist_count"])  # start with existing series
        # lag_1 is the most recent count
        lag_1 = recent_counts[-1]
        # lag_12 is 12 months back if available, else use the first available count
        lag_12 = recent_counts[-12] if len(recent_counts) >= 12 else recent_counts[0]
        feature["tourist_lag_1"] = lag_1
        feature["tourist_lag_12"] = lag_12
        # Convert to DataFrame for prediction
        pred_df = pd.DataFrame([feature])
        pred = pipe.predict(pred_df)[0]
        forecasts.append({"date": current_date, "predicted_tourists": int(round(pred))})
        # Update recent_counts and last_row for next iteration (use predicted count as previous_month_tourists)
        recent_counts.append(pred)
        last_row["tourist_count"] = pred
    return pd.DataFrame(forecasts)
