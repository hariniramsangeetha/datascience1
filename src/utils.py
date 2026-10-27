import joblib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_model(model: Any, model_path: str) -> None:
    """Save a trained model to the given path using joblib.

    Args:
        model: Trained model object.
        model_path: Destination file path (including .pkl extension).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str) -> Any:
    """Load a model from a joblib file.

    Args:
        model_path: Path to the .pkl file.
    Returns:
        The deserialized model.
    """
    return joblib.load(model_path)


def save_metadata(metadata: Dict[str, Any], metadata_path: str) -> None:
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE (avoid division by zero)
    non_zero = y_true != 0
    mape = (np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]).mean() * 100 if np.any(non_zero) else np.nan)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
