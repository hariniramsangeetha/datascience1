import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_FILE = DATA_DIR / "tourism_data.csv"
PROCESSED_FILE = DATA_DIR / "processed_data.csv"

DESTINATIONS = [
    "Ooty",
    "Kodaikanal",
    "Munnar",
    "Goa",
    "Jaipur",
    "Mysuru",
    "Chennai",
    "Madurai",
    "Kerala",
    "Coorg",
]

def generate_synthetic_data(num_records: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic tourism dataset.

    The function creates *num_records* rows covering several years, months,
    and the destinations defined in ``DESTINATIONS``. Relationships are
    approximated so that the data looks plausible for demonstration purposes.
    """
    np.random.seed(seed)
    # Create a date range covering ~4 years (48 months) for each destination
    start_date = datetime(2018, 1, 1)
    dates = []
    for i in range(num_records):
        # Random month/year within 2018-2023
        offset_months = np.random.randint(0, 60)  # up to 5 years
        year = 2018 + offset_months // 12
        month = (offset_months % 12) + 1
        day = np.random.randint(1, 28)  # avoid month end issues
        dates.append(datetime(year, month, day))

    data = pd.DataFrame({"date": dates})
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["destination"] = np.random.choice(DESTINATIONS, size=num_records)

    # Base tourist count per destination (higher for popular places)
    base_pop = {
        "Ooty": 1200,
        "Kodaikanal": 1100,
        "Munnar": 1000,
        "Goa": 2500,
        "Jaipur": 2000,
        "Mysuru": 1500,
        "Chennai": 3000,
        "Madurai": 1800,
        "Kerala": 2200,
        "Coorg": 900,
    }

    # Seasonal multiplier (higher in peak months)
    month_multiplier = {
        1: 0.8,
        2: 0.9,
        3: 1.0,
        4: 1.2,
        5: 1.4,
        6: 1.5,
        7: 1.6,
        8: 1.5,
        9: 1.3,
        10: 1.1,
        11: 0.9,
        12: 0.8,
    }

    # Weather generation – simple sinusoidal patterns plus random noise
    def generate_weather(row):
        # Temperature: cooler in hill stations, warmer in coastal
        temp_base = 20 if row["destination"] in ["Ooty", "Kodaikanal", "Munnar", "Coorg"] else 30
        # Seasonal variation +/-5°C
        temp = temp_base + 5 * np.sin((row["month"] - 1) / 12 * 2 * np.pi) + np.random.normal(0, 1.5)
        # Rainfall: higher in monsoon months (June‑Sept) for most places
        rain = max(0, np.random.normal(80 if 6 <= row["month"] <= 9 else 20, 15))
        # Humidity correlated with rainfall
        humidity = min(100, max(30, rain * 0.6 + np.random.normal(50, 10)))
        return pd.Series([temp, rain, humidity])

    data[["temperature", "rainfall", "humidity"]] = data.apply(generate_weather, axis=1)

    # Holiday & festival flags – higher probability during certain months
    holiday_months = {12, 1, 5, 8}
    festival_months = {3, 10, 11}
    data["holiday"] = data["month"].isin(holiday_months).astype(int)
    data["festival"] = data["month"].isin(festival_months).astype(int)
    data["weekend"] = data["date"].dt.weekday.isin([5, 6]).astype(int)

    # Hotel price – higher in peak season, varies by destination
    price_base = {
        "Ooty": 120,
        "Kodaikanal": 110,
        "Munnar": 115,
        "Goa": 150,
        "Jaipur": 130,
        "Mysuru": 125,
        "Chennai": 140,
        "Madurai": 130,
        "Kerala": 135,
        "Coorg": 115,
    }
    data["hotel_price"] = data["destination"].map(price_base) * month_multiplier.get(data["month"].iloc[0], 1)
    # Add some random noise to price
    data["hotel_price"] = data["hotel_price"] * (1 + np.random.normal(0, 0.05, size=num_records))
    data["hotel_price"] = data["hotel_price"].round(2)

    # Base tourist count influenced by destination, month, holiday, festival, weather
    base = data["destination"].map(base_pop)
    seasonal = data["month"].map(month_multiplier)
    holiday_effect = 1 + 0.25 * data["holiday"]
    festival_effect = 1 + 0.15 * data["festival"]
    rain_effect = np.exp(-0.003 * data["rainfall"])  # heavy rain reduces demand
    temp_effect = 1 + 0.005 * (data["temperature"] - 25)  # slight temp influence
    noise = np.random.normal(0, 100, size=num_records)
    tourists = base * seasonal * holiday_effect * festival_effect * rain_effect * temp_effect + noise
    data["tourist_count"] = tourists.clip(lower=0).round().astype(int)

    # Hotel occupancy roughly proportional to tourist count (capacity factor)
    capacity_factor = 0.6 + 0.3 * np.random.rand(num_records)  # 60‑90% typical
    data["hotel_occupancy"] = (data["tourist_count"] * capacity_factor / 100).clip(0, 100).round(2)

    # Transport availability – higher when more tourists
    data["transport_availability"] = (np.random.uniform(0.7, 1.0, size=num_records) * (1 + 0.1 * data["tourist_count"] / data["tourist_count"].max()))
    data["transport_availability"] = data["transport_availability"].round(2)

    # Previous month tourists – will be filled later after sorting
    data = data.sort_values("date").reset_index(drop=True)
    data["previous_month_tourists"] = data["tourist_count"].shift(1).fillna(data["tourist_count"].median())

    # Reorder columns
    cols = [
        "date",
        "year",
        "month",
        "destination",
        "tourist_count",
        "temperature",
        "rainfall",
        "humidity",
        "holiday",
        "festival",
        "weekend",
        "hotel_price",
        "hotel_occupancy",
        "transport_availability",
        "previous_month_tourists",
    ]
    data = data[cols]
    return data

def save_raw_data(df: pd.DataFrame) -> None:
    """Write the raw synthetic dataframe to CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)

def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV (generate if missing)."""
    if not RAW_FILE.exists():
        df = generate_synthetic_data()
        save_raw_data(df)
    else:
        df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform cleaning steps and return a processed dataframe.
    Steps include:
    - Missing value handling (simple imputation)
    - Duplicate removal
    - Ensure correct dtypes
    - Basic outlier clipping for numeric columns
    """
    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Identify numeric columns (excluding date)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Simple missing value handling: fill numeric NaNs with median, categorical with mode
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    if df["destination"].isna().any():
        df["destination"] = df["destination"].fillna(df["destination"].mode()[0])

    # Clip extreme outliers to 1st/99th percentiles
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # Ensure proper dtypes
    df["holiday"] = df["holiday"].astype(int)
    df["festival"] = df["festival"].astype(int)
    df["weekend"] = df["weekend"].astype(int)

    # Save processed version
    df.to_csv(PROCESSED_FILE, index=False)
    return df

def load_processed_data() -> pd.DataFrame:
    """Load processed data, generating and preprocessing if necessary."""
    if not PROCESSED_FILE.exists():
        raw = load_raw_data()
        processed = preprocess_data(raw)
        return processed
    else:
        return pd.read_csv(PROCESSED_FILE, parse_dates=["date"])

def data_quality_report(df: pd.DataFrame) -> str:
    """Return a concise summary string of data quality metrics."""
    total = len(df)
    missing = df.isna().sum().sum()
    duplicates = df.duplicated().sum()
    features = df.shape[1]
    destinations = df["destination"].nunique()
    return (
        f"Total Records: {total:,}\n"
        f"Missing Values: {missing}\n"
        f"Duplicate Records: {duplicates}\n"
        f"Features: {features}\n"
        f"Destinations: {destinations}\n"
    )

if __name__ == "__main__":
    # Quick generation/check when run directly
    df = load_processed_data()
    print(data_quality_report(df))
