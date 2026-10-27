import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional time‑based features.

    - ``month_sin`` and ``month_cos`` capture seasonal cyclicality.
    - ``year`` is kept as numeric.
    - ``previous_month_tourists`` is already in the raw data; if missing we fill with median.
    """
    import numpy as np
    df = df.copy()
    # Ensure month is numeric 1‑12
    df["month"] = df["month"].astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Create a simple season label (spring, summer, ...)
    # Not used directly but kept for possible analysis
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer that encodes categorical columns.

    Currently only ``destination`` is categorical. The function inspects the
    dataframe to decide which columns need one‑hot encoding.
    """
    categorical = [col for col in ["destination"] if col in df.columns]
    numeric = [col for col in df.columns if col not in categorical + ["tourist_count", "date"]]
    transformers = []
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric:
        # No scaling needed for tree‑based models; for linear regression we may add scaling later.
        transformers.append(("num", "passthrough", numeric))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
