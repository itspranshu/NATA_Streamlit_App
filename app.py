# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# -------------------------
# File locations (check repo root first, then Desktop fallback)
# -------------------------
# Repo-root (recommended for GitHub + Streamlit Cloud)
MODEL_FILE_LOCAL = Path("./nata_model.pkl")
FEATURES_FILE_LOCAL = Path("./nata_features.json")
DEFAULTS_FILE_LOCAL = Path("./default_values.json")

# Desktop fallback (useful for local testing if files are on Desktop)
DESKTOP = Path.home() / "OneDrive" / "Desktop"
MODEL_FILE_DESKTOP = DESKTOP / "nata_model.pkl"
FEATURES_FILE_DESKTOP = DESKTOP / "nata_features.json"
DEFAULTS_FILE_DESKTOP = DESKTOP / "default_values.json"

def find_file(local_path: Path, desktop_path: Path) -> Path | None:
    if local_path.exists():
        return local_path
    if desktop_path.exists():
        return desktop_path
    return None

MODEL_PATH = find_file(MODEL_FILE_LOCAL, MODEL_FILE_DESKTOP)
FEATURES_PATH = find_file(FEATURES_FILE_LOCAL, FEATURES_FILE_DESKTOP)
DEFAULTS_PATH = find_file(DEFAULTS_FILE_LOCAL, DEFAULTS_FILE_DESKTOP)

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="🛒 NATA Supermarket – Customer Insights App", layout="centered")
st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")

# -------------------------
# Load model, feature list, defaults
# -------------------------
model = None
feature_cols = None
default_values = {}

if MODEL_PATH is None:
    st.error("❌ Model file not found. Put `nata_model.pkl` in repo root or on your Desktop.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        st.error(f"❌ Failed to load model from {MODEL_PATH}: {e}")

if FEATURES_PATH is None:
    st.error("❌ Feature list file not found. Put `nata_features.json` in repo root or on your Desktop.")
else:
    try:
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        st.success(f"📄 Feature list loaded from: {FEATURES_PATH}")
    except Exception as e:
        st.error(f"❌ Failed to load feature list: {e}")

if DEFAULTS_PATH is None:
    st.warning("⚠️ default_values.json not found. The app will use zero/empty defaults if needed.")
else:
    try:
        with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
            default_values = json.load(f)
        st.success(f"📥 Default values loaded from: {DEFAULTS_PATH}")
    except Exception as e:
        st.warning(f"⚠️ Could not read default_values.json: {e}. The app will use zero/empty defaults.")

# -------------------------
# CHOSEN UI INPUT VARIABLES (7)
# -------------------------
# These are the 7 you asked for — user will enter these, the rest are filled from defaults.
chosen_inputs = [
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumWebPurchases",
    "NumStorePurchases",
    "Education"   # encoded category (int)
]

st.header("📥 Enter Customer Details (only 7 fields)")

# Build UI controls using defaults when present
ui_values = {}
for feat in chosen_inputs:
    # If default exists in default_values, use it as the default for the input widget
    default_val = default_values.get(feat, None)
    # Categorical Education -> use selectbox of integers (assume encoding 0..6; adjust if needed)
    if feat == "Education":
        # If default is string, attempt to coerce to int; otherwise use 0
        try:
            default_int = int(default_val) if default_val is not None else 0
        except Exception:
            default_int = 0
        ui_values[feat] = st.selectbox(
            "Education (encoded category)",
            options=[0,1,2,3,4,5,6],
            index=0 if default_int not in [0,1,2,3,4,5,6] else [0,1,2,3,4,5,6].index(default_int)
        )
    else:
        # For numeric fields use number_input
        if default_val is None:
            default_val = 0.0
        # ensure numeric
        try:
            dv = float(default_val)
        except Exception:
            dv = 0.0
        ui_values[feat] = st.number_input(feat, value=dv)

st.subheader("🔍 Input Preview (these 7 values will be used; other features are filled automatically)")
st.dataframe(pd.DataFrame([ui_values]))

# -------------------------
# Prepare prediction row
# -------------------------
def build_prediction_row(feature_cols, user_inputs, defaults):
    """
    Build one-row DataFrame in exact order of feature_cols.
    - Start from defaults (if available)
    - Update with user_inputs
    - Ensure types are Python-native
    """
    row = {}
    for col in feature_cols:
        if col in user_inputs:
            val = user_inputs[col]
        else:
            # fall back to defaults, then to safe zero
            val = defaults.get(col, 0.0)
        # convert numpy types to native python
        if isinstance(val, (np.integer,)):
            val = int(val)
        if isinstance(val, (np.floating,)):
            val = float(val)
        row[col] = val
    return pd.DataFrame([row], columns=feature_cols)

# -------------------------
# Prediction button
# -------------------------
if st.button("🔮 Predict Customer Spending (using model)"):
    if model is None or feature_cols is None:
        st.error("Model or feature list not loaded; cannot predict.")
    else:
        try:
            # Build full input row for the model using feature_cols ordering
            pred_row = build_prediction_row(feature_cols, ui_values, default_values)

            # Convert any categorical encodings to numeric types if needed
            # (the model pipeline's preprocessor will handle OneHotEncoder)
            for c in pred_row.columns:
                # try to coerce numeric columns to numeric dtype
                try:
                    pred_row[c] = pd.to_numeric(pred_row[c])
                except Exception:
                    pass

            st.write("Final row sent to model:")
            st.dataframe(pred_row.T.rename(columns={0: "value"}))

            # Run prediction
            prediction = model.predict(pred_row)[0]

            st.success(f"💰 Predicted Total Spending: ₹{prediction:,.2f}")

            # Basic recommendation
            if prediction > 5:
                st.info("🛍 High-value customer — recommend premium/up-sell & loyalty offers.")
            elif prediction > 2:
                st.info("🛒 Mid-value customer — target with moderate discounts & combos.")
            else:
                st.info("📌 Low-value — focus on awareness campaigns, bundles.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------
# Footer / credits
# -------------------------
st.markdown("---")
st.markdown(
    "**Developed by Prashant Singh (IIM Sirmaur)**  \n"
    "Model Training ▪ Clustering Analysis ▪ Hyperparameter Tuning ▪ Streamlit Deployment"
)
