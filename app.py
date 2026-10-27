import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# Import internal modules
from src import data_preprocessing, train_model, prediction, forecast, eda, utils

st.set_page_config(page_title="🌍 TourismPulse", layout="wide")

# -----------------------------------------------------------------------------
# Load or generate data
# -----------------------------------------------------------------------------
processed_path = Path(__file__).resolve().parents[1] / "data" / "processed_data.csv"
if not processed_path.exists():
    # This will generate raw data and process it automatically
    df = data_preprocessing.load_processed_data()
else:
    df = pd.read_csv(processed_path, parse_dates=["date"])

# -----------------------------------------------------------------------------
# Load or train model
# -----------------------------------------------------------------------------
model_path = Path(__file__).resolve().parents[1] / "models" / "tourism_model.pkl"
if not model_path.exists():
    train_model.train_and_select_model()
    best_pipe = utils.load_model(str(model_path))
else:
    best_pipe = utils.load_model(str(model_path))

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🗺️ TourismPulse Dashboard")
section = st.sidebar.radio("Navigate", [
    "Home",
    "Destination Analytics",
    "Tourism Trends",
    "Forecasting",
    "What‑If Simulator",
    "Model Performance",
    "Data Explorer",
])

# -----------------------------------------------------------------------------
# Home page – KPI cards and overview plots
# -----------------------------------------------------------------------------
if section == "Home":
    st.title("🌍 TourismPulse")
    st.subheader("Intelligent Tourist Footfall Forecasting & Destination Analytics")
    kpis = eda.kpi_cards(df)
    cols = st.columns(len(kpis))
    for col, (label, value) in zip(cols, kpis.items()):
        col.metric(label, value)
    st.markdown("---")
    st.plotly_chart(eda.tourist_trend(df), use_container_width=True)
    st.plotly_chart(eda.destination_popularity(df), use_container_width=True)
    st.markdown("---")
    st.subheader("Key Insights")
    top_dest = kpis["Top Destination"]
    peak_month = kpis["Peak Tourism Month"]
    st.write(f"• **{top_dest}** records the highest total visitor count.")
    st.write(f"• The month with the greatest footfall is **{peak_month}**.")
    st.write("• Heavy rainfall generally reduces tourist numbers, while holidays boost them.")

# -----------------------------------------------------------------------------
# Destination Analytics page
# -----------------------------------------------------------------------------
elif section == "Destination Analytics":
    st.title("Destination Analytics")
    dest = st.selectbox("Select Destination", sorted(df["destination"].unique()))
    sub = df[df["destination"] == dest]
    total_visitors = int(sub["tourist_count"].sum())
    avg_monthly = round(sub["tourist_count"].mean(), 2)
    peak_month = sub.groupby("month")["tourist_count"].sum().idxmax()
    low_month = sub.groupby("month")["tourist_count"].sum().idxmin()
    growth = (sub.groupby("year")["tourist_count"].sum().pct_change().iloc[-1] * 100) if len(sub["year"].unique()) > 1 else 0
    avg_occupancy = round(sub["hotel_occupancy"].mean(), 2)
    col1, col2 = st.columns(2)
    col1.metric("Total Visitors", f"{total_visitors:,}")
    col2.metric("Average Monthly Visitors", f"{avg_monthly:,}")
    st.metric("Peak Month (by visitors)", peak_month)
    st.metric("Lowest Month", low_month)
    st.metric("Year‑over‑Year Growth", f"{growth:.2f}%")
    st.metric("Avg. Hotel Occupancy", f"{avg_occupancy}%")
    st.subheader("Visitor Trend for " + dest)
    st.plotly_chart(eda.tourist_trend(sub), use_container_width=True)
    st.subheader("Weather Impact")
    st.plotly_chart(eda.weather_vs_tourists(sub), use_container_width=True)

# -----------------------------------------------------------------------------
# Tourism Trends page – overall visualisations
# -----------------------------------------------------------------------------
elif section == "Tourism Trends":
    st.title("Tourism Trends")
    st.subheader("Overall Tourist Footfall Trend")
    st.plotly_chart(eda.tourist_trend(df), use_container_width=True)
    st.subheader("Destination Popularity")
    st.plotly_chart(eda.destination_popularity(df), use_container_width=True)
    st.subheader("Weather vs Tourist Count")
    st.plotly_chart(eda.weather_vs_tourists(df), use_container_width=True)

# -----------------------------------------------------------------------------
# Forecasting page
# -----------------------------------------------------------------------------
elif section == "Forecasting":
    st.title("Future Tourist Footfall Forecast")
    dest = st.selectbox("Destination", sorted(df["destination"].unique()))
    periods = st.slider("Forecast horizon (months)", 1, 12, 6)
    forecast_df = forecast.forecast_future(df, dest, periods)
    st.subheader(f"Forecast for {dest} – Next {periods} months")
    st.dataframe(forecast_df.style.format({"predicted_tourists": "{:,}"}))
    fig = px.line(forecast_df, x="date", y="predicted_tourists", markers=True, title="Forecast Trend")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# What‑If Simulator page
# -----------------------------------------------------------------------------
elif section == "What‑If Simulator":
    st.title("What‑If Tourism Simulator")
    st.write("Adjust the sliders below to see how changes affect the predicted tourist count.")
    dest = st.selectbox("Destination", sorted(df["destination"].unique()))
    medians = df.median(numeric_only=True)
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", 0.0, 40.0, float(medians["temperature"]))
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, float(medians["rainfall"]))
        humidity = st.slider("Humidity (%)", 20.0, 100.0, float(medians["humidity"]))
    with col2:
        holiday = st.checkbox("Holiday", False)
        festival = st.checkbox("Festival", False)
        hotel_price = st.slider("Hotel Price (USD)", 50, 300, int(medians["hotel_price"]))
        hotel_occupancy = st.slider("Hotel Occupancy (%)", 0, 100, int(medians["hotel_occupancy"]))
    input_dict = {
        "destination": dest,
        "year": pd.Timestamp.now().year,
        "month": pd.Timestamp.now().month,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity,
        "holiday": int(holiday),
        "festival": int(festival),
        "weekend": int(pd.Timestamp.now().weekday() >= 5),
        "hotel_price": hotel_price,
        "hotel_occupancy": hotel_occupancy,
        "transport_availability": float(medians["transport_availability"]),
        "previous_month_tourists": float(medians["previous_month_tourists"]),
    }
    input_dict["month_sin"] = np.sin(2 * np.pi * input_dict["month"] / 12)
    input_dict["month_cos"] = np.cos(2 * np.pi * input_dict["month"] / 12)
    # Add lag features expected by the model
    input_dict["tourist_lag_1"] = input_dict["previous_month_tourists"]
    input_dict["tourist_lag_12"] = input_dict["previous_month_tourists"]
    baseline_dict = dict(input_dict)
    baseline_dict["holiday"] = 0
    baseline_dict["festival"] = 0
    baseline_pred = prediction.predict_single(baseline_dict)
    scenario_pred = prediction.predict_single(input_dict)
    change_pct = ((scenario_pred - baseline_pred) / baseline_pred) * 100 if baseline_pred != 0 else 0
    st.subheader("Prediction Results")
    st.write(f"**Baseline prediction** (current conditions): {baseline_pred:,.0f} tourists")
    st.write(f"**Scenario prediction**: {scenario_pred:,.0f} tourists")
    st.write(f"**Percent change**: {change_pct:+.2f}%")

# -----------------------------------------------------------------------------
# Model Performance page
# -----------------------------------------------------------------------------
elif section == "Model Performance":
    st.title("Model Performance Comparison")
    meta_path = Path(__file__).resolve().parents[1] / "models" / "model_metadata.json"
    if meta_path.exists():
        meta = utils.load_metadata(str(meta_path))
        df_metrics = pd.DataFrame(meta["metrics"]).T
        st.dataframe(df_metrics.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "MAPE": "{:.2f}%"}))
        st.success(f'Best model selected: **{meta["best_model"]}**')
        model = utils.load_model(str(model_path)).named_steps["model"]
        # Load model_path
        model_path = Path(__file__).resolve().parents[1] / "models" / "tourism_model.pkl"
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance (tree‑based model)")
            feature_names = list(best_pipe.named_steps["preproc"].get_feature_names_out())
            importances = model.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False)
            fig = px.bar(imp_df.head(15), x="importance", y="feature", orientation="h", title="Top 15 Feature Importances")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selected model does not provide feature importance.")
    else:
        st.warning("Model metadata not found. Run training first.")

# -----------------------------------------------------------------------------
# Data Explorer page – upload & preview
# -----------------------------------------------------------------------------
elif section == "Data Explorer":
    st.title("Data Explorer")
    uploaded = st.file_uploader("Upload your own tourism CSV file", type=["csv"])
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded, parse_dates=["date"])
            st.success("File uploaded successfully!")
            st.subheader("Preview (first 10 rows)")
            st.dataframe(user_df.head(10))
            required = {"date", "year", "month", "destination", "tourist_count", "temperature", "rainfall", "humidity", "holiday", "festival", "weekend", "hotel_price", "hotel_occupancy", "transport_availability", "previous_month_tourists"}
            missing = required - set(user_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.success("All required columns are present.")
                if st.button("Use this dataset for analysis"):
                    user_df.to_csv(str(Path(__file__).resolve().parents[1] / "data" / "processed_data.csv"), index=False)
                    st.experimental_rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a CSV matching the required schema to explore your data.")
        st.subheader("Current synthetic dataset summary")
        st.text(eda.data_quality_report(df))

# End of app
