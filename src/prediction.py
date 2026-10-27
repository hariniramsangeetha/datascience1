import pandas as pd
from pathlib import Path

from .utils import load_model, compute_metrics

# Load the best pipeline once at import time (lazy loading could be added later)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tourism_model.pkl"

if MODEL_PATH.exists():
    _best_pipeline = load_model(str(MODEL_PATH))
else:
    _best_pipeline = None


def predict_single(input_dict: dict) -> float:
    """Predict tourist count for a single observation.

    ``input_dict`` should contain the same feature keys that were used during training
    (excluding the target ``tourist_count`` and ``date``). The function converts the
    dictionary into a single‑row DataFrame, runs the stored pipeline and returns the
    predicted value as a float.
    """
    if _best_pipeline is None:
        raise FileNotFoundError("Trained model not found. Please run train_model.py first.")
    df = pd.DataFrame([input_dict])
    # Ensure column ordering matches training data (the pipeline will handle it)
    pred = _best_pipeline.predict(df)
    return float(pred[0])


def batch_predict(df: pd.DataFrame) -> pd.Series:
    """Predict tourist counts for a DataFrame of input features.
    Returns a pandas Series of predictions.
    """
    if _best_pipeline is None:
        raise FileNotFoundError("Trained model not found.")
    preds = _best_pipeline.predict(df)
    return pd.Series(preds, index=df.index)
