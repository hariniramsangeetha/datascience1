import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def load_data(processed_path: str = None) -> pd.DataFrame:
    """Load the processed dataset.
    If *processed_path* is None, locate the default CSV inside the project.
    """
    if processed_path is None:
        base_dir = Path(__file__).resolve().parents[2]
        processed_path = base_dir / "data" / "processed_data.csv"
    return pd.read_csv(processed_path, parse_dates=["date"])


def kpi_cards(df: pd.DataFrame) -> dict:
    """Calculate KPI values used on the home page.
    Returns a dictionary with keys:
    - total_visits
    - top_destination
    - peak_month
    - forecast_next_month (placeholder)
    - avg_hotel_occupancy
    """
    total_visits = int(df["tourist_count"].sum())
    top_destination = df.groupby("destination")["tourist_count"].sum().idxmax()
    peak_month = df.groupby("month")["tourist_count"].sum().idxmax()
    avg_hotel_occupancy = round(df["hotel_occupancy"].mean(), 2)
    return {
        "Total Tourist Visits": total_visits,
        "Top Destination": top_destination,
        "Peak Tourism Month": peak_month,
        "Average Hotel Occupancy": f"{avg_hotel_occupancy}%",
    }


def tourist_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of tourist count over time (aggregated monthly)."""
    df_month = df.resample("M", on="date").sum().reset_index()
    fig = px.line(df_month, x="date", y="tourist_count", title="Tourist Footfall Trend")
    fig.update_layout(xaxis_title="Date", yaxis_title="Visitors")
    return fig


def destination_popularity(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total visitors per destination."""
    agg = df.groupby("destination")["tourist_count"].sum().reset_index().sort_values("tourist_count", ascending=False)
    fig = px.bar(agg, x="destination", y="tourist_count", title="Destination Popularity")
    fig.update_layout(xaxis_title="Destination", yaxis_title="Total Visitors")
    return fig


def weather_vs_tourists(df: pd.DataFrame) -> go.Figure:
    """Scatter matrix of temperature, rainfall, humidity vs tourist count.
    Returns a combined figure with three subplots.
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Temp vs Tourists", "Rainfall vs Tourists", "Humidity vs Tourists"])
    fig.add_trace(go.Scatter(x=df["temperature"], y=df["tourist_count"], mode="markers", opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["rainfall"], y=df["tourist_count"], mode="markers", opacity=0.6), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["humidity"], y=df["tourist_count"], mode="markers", opacity=0.6), row=1, col=3)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Rainfall (mm)", row=1, col=2)
    fig.update_xaxes(title_text="Humidity (%)", row=1, col=3)
    fig.update_yaxes(title_text="Tourist Count", row=1, col=1)
    fig.update_layout(title="Weather Impact on Tourist Footfall", showlegend=False)
    return fig

from plotly.subplots import make_subplots


def data_quality_report(df: pd.DataFrame) -> str:
    """Return a plain-text summary of dataset shape, types, nulls, and basic stats."""
    lines = []
    lines.append(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    lines.append("")
    lines.append("Column Types:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        lines.append(f"  {col:<30s}  {str(df[col].dtype):<12s}  nulls: {null_count}")
    lines.append("")
    lines.append("Numeric Summary:")
    desc = df.describe(include='number').T
    for col in desc.index:
        row = desc.loc[col]
        lines.append(f"  {col:<30s}  mean={row['mean']:<10.2f}  std={row['std']:<10.2f}  min={row['min']:<10.2f}  max={row['max']:<10.2f}")
    return "\n".join(lines)
