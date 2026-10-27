# TourismPulse

## Project Overview

**TourismPulse – Intelligent Tourist Footfall Forecasting & Destination Analytics** is a Python‑based Data Science dashboard built with Streamlit. It demonstrates a full analytics workflow:

1. **Synthetic data generation** (or user‑uploaded CSV)
2. **Data cleaning & validation**
3. **Exploratory Data Analysis** with interactive Plotly charts
4. **Feature engineering** for regression and time‑series models
5. **Model training** – Linear Regression, Random Forest, Gradient Boosting/XGBoost
6. **Model evaluation** and automatic best‑model selection
7. **Forecasting** – future tourist demand per destination
8. **What‑If simulation** – explore how weather, holidays, price, etc. affect demand
9. **Destination ranking** and automated insights

The dashboard is launched with:

```bash
streamlit run app.py
```

All source code lives under the `src/` package; raw and processed CSV files are stored in `data/`; the best model is saved under `models/`.

---
### Quick Start (Windows)

```powershell
# Clone / copy the project folder
cd "C:\Users\harin\OneDrive\Documents\Tourist Footfall Forecasting and Destination Analytics\TourismPulse"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

*Linux/macOS equivalents are provided in the README.

---
### Folder Structure

```
TourismPulse/
│
├─ app.py                     # Streamlit entry point
├─ requirements.txt
├─ .gitignore
├─ README.md
│
├─ data/
│   ├─ tourism_data.csv       # Synthetic (or user‑uploaded) raw data
│   └─ processed_data.csv     # Cleaned version after preprocessing
│
├─ models/
│   ├─ tourism_model.pkl      # Serialized best model
│   └─ model_metadata.json    # Performance metrics & training info
│
├─ notebooks/
│   └─ tourism_analysis.ipynb # Reference notebook for developers
│
├─ src/
│   ├─ __init__.py
│   ├─ data_preprocessing.py
│   ├─ feature_engineering.py
│   ├─ eda.py
│   ├─ train_model.py
│   ├─ prediction.py
│   ├─ forecast.py
│   └─ utils.py
│
└─ visualizations/            # Optional saved Plotly JSON / images
```

---
### License

MIT License – feel free to adapt, extend, and use for educational purposes.
