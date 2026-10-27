import sys, pandas as pd
sys.path.append(r"C:/Users/harin/OneDrive/Documents/Tourist Footfall Forecasting and Destination Analytics/TourismPulse")
from src.forecast import forecast_future

df = pd.read_csv(r"C:/Users/harin/OneDrive/Documents/Tourist Footfall Forecasting and Destination Analytics/data/processed_data.csv", parse_dates=['date'])

destination = df['destination'].iloc[0]

result = forecast_future(df, destination, periods=6)
print(result)
