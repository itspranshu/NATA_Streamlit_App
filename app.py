# app.py (paste this into your repo / replace current app.py)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from datetime import datetime
from pathlib import Path

# ---------------------------
# Config - change if required
# ---------------------------
DESKTOP = Path(r"C:\Users\psngh\OneDrive\Desktop")
MODEL_FILE = DESKTOP / "nata_model.pkl"
FEATURE_FILE = DESKTOP / "nata_features.json"
DEFAULTS_FILE = DESKTOP / "default_values.json"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="🛒 NATA Supermarket – Customer Insights App", layout="centered")
st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")

# ---------------------------
# Load model & metadata
# ---------------------------
model = None
feature_cols = None
default_values = {}

if MODEL_FILE.exists():
    model = joblib.load(MODEL_FILE)
    st.success(f"✅ Model loaded successfully from Desktop.")
else:
    st.error(f"❌ Model file not found at Desktop ({MODEL_FILE}). Upload it to the app folder or Desktop.")

if FEATURE_FILE.exists():
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    st.success("📄 Feature list loaded successfully.")
else:
    st.error(f"❌ Feature list not found at Desktop ({FEATURE_FILE}).")

if DEFAULTS_FILE.exists():
    with open(DEFAULTS_FILE, "r", encoding="utf-8") as f:
        default_values = json.load(f)
    st.success("🔧 Default values loaded.")
else:
    st.warning(f"⚠️ default_values.json not found at Desktop ({DEFAULTS_FILE}). App will still try to run but may use zeros for missing features.")

st.markdown("---")
st.header("📥 Enter Customer Details (only 7 inputs)")

# ---------------------------
# The 7 user-friendly inputs
# ---------------------------
# 1) Income
income = st.number_input("Income (annual, ₹)", min_value=0.0, value=float(default_values.get("Income", 0.0)), step=1000.0, format="%.2f")

# 2) Age (years) — we'll convert to Year_Birth internally
age = st.number_input("Age (years)", min_value=16, max_value=120, value=int( (datetime.now().year - float(default_values.get("Year_Birth", datetime.now().year-35))) if default_values.get("Year_Birth") else 30 ))

# 3) Education - friendly labels (maps to the dataset's textual categories)
education = st.selectbox("Education", options=["Graduation", "PhD", "Basic", "Master"], index=0)

# 4) Marital status
marital = st.selectbox("Marital status", options=["Single", "Married", "Together", "Divorced", "Widow"], index=0)

# 5) Recency (days)
recency = st.number_input("Recency (days since last purchase)", min_value=0, value=int(default_values.get("Recency", 30)))

# 6) NumStorePurchases
num_store = st.number_input("Number of store purchases", min_value=0, value=int(default_values.get("NumStorePurchases", 0)))

# 7) NumWebPurchases
num_web = st.number_input("Number of web purchases", min_value=0, value=int(default_values.get("NumWebPurchases", 0)))

# ---------------------------
# Build final input vector for the model
# ---------------------------
if feature_cols is None:
    st.stop()  # nothing to do

# Convert Age -> Year_Birth (model expects Year_Birth)
current_year = datetime.now().year
year_birth = current_year - int(age)

# Collect user-provided values in a dict keyed by feature name
user_inputs = {
    "Income": float(income),
    "Year_Birth": int(year_birth),
    "Education": str(education),
    "Marital_Status": str(marital),
    "Recency": int(recency),
    "NumStorePurchases": int(num_store),
    "NumWebPurchases": int(num_web),
}

# Merge: for each feature in feature_cols, prefer user input, otherwise default_values, otherwise safe fallback
final_input = {}
for feat in feature_cols:
    if feat in user_inputs:
        final_input[feat] = user_inputs[feat]
    else:
        # Try default_values (json stores Python types)
        if feat in default_values:
            final_input[feat] = default_values[feat]
        else:
            # Last resort fallback type inference: numeric -> 0, categorical -> "".
            # We'll check feature name heuristically:
            if feat.lower() in ["education","marital_status"]:
                final_input[feat] = ""
            else:
                final_input[feat] = 0

# Ensure datatypes match: convert numpy types to native python types
for k, v in list(final_input.items()):
    if isinstance(v, (np.integer,)):
        final_input[k] = int(v)
    if isinstance(v, (np.floating,)):
        final_input[k] = float(v)

# Present input preview
st.subheader("🔎 Input preview (what the model receives)")
preview_df = pd.DataFrame([final_input])
st.dataframe(preview_df.T.rename(columns={0:"value"}))

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict Customer Spending"):
    if model is None:
        st.error("Model not loaded.")
    else:
        # Create DataFrame in the exact column order feature_cols
        X_pred = pd.DataFrame([final_input], columns=feature_cols)
        try:
            pred = model.predict(X_pred)[0]
            st.success(f"💰 Predicted Total Spending: ₹{pred:,.2f}")
            # Simple recommendations
            if pred > 5000:
                st.info("🛍 High-value customer: target with premium promotions & loyalty offers.")
            elif pred > 2000:
                st.info("🛒 Mid-value: cross-sell and bundle offers could work.")
            else:
                st.info("📌 Lower-value: awareness campaigns and entry discounts.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("**Developed by Prashant Singh (IIM Sirmaur)**  \nModel Training ▪ Clustering Analysis ▪ Hyperparameter Tuning ▪ Streamlit Deployment")
