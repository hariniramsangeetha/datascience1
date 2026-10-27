import pandas as pd
from src.forecast import forecast_future

# Load processed data (adjust path as needed)
csv_path = r"C:/Users/harin/OneDrive/Documents/Tourist Footfall Forecasting and Destination Analytics/data/processed_data.csv"

df = pd.read_csv(csv_path, parse_dates=['date'])

# Choose a destination present in data
destination = df['destination'].iloc[0]

# Generate 6‑month forecast
result = forecast_future(df, destination, periods=6)
print(result)
