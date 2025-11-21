import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ======================================================
# File paths (load directly from repository)
# ======================================================
MODEL_FILE = "nata_model.pkl"
FEATURE_FILE = "nata_features.json"
DEFAULT_FILE = "default_values.json"

# ======================================================
# Page setup
# ======================================================
st.set_page_config(page_title="🛒 NATA Supermarket – Customer Insights", layout="centered")
st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")

# ======================================================
# Load model
# ======================================================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found.")
    st.stop()

# Load feature list
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, "r") as f:
        feature_cols = json.load(f)
else:
    st.error("❌ Feature file not found.")
    st.stop()

# Load default values
if os.path.exists(DEFAULT_FILE):
    with open(DEFAULT_FILE, "r") as f:
        default_values = json.load(f)
else:
    st.error("❌ default_values.json not found.")
    st.stop()

# ======================================================
# Only 7 variables for user input
# ======================================================
user_input_vars = [
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

st.header("📥 Enter Customer Details")

user_inputs = {}

for var in user_input_vars:
    user_inputs[var] = st.number_input(var, value=0.0)

# ======================================================
# Build final input row by merging:
#  7 user inputs +
#  12 default values
# ======================================================
final_input = {}

for col in feature_cols:
    if col in user_input_vars:
        final_input[col] = user_inputs[col]
    else:
        final_input[col] = default_values[col]

input_df = pd.DataFrame([final_input])

st.subheader("🔍 Final Input Used by Model")
st.dataframe(input_df)

# ======================================================
# Predict
# ======================================================
if st.button("🔮 Predict Customer Spending"):
    prediction = model.predict(input_df)[0]

    st.success(f"💰 **Predicted Total Spending: ₹{prediction:,.2f}**")

    if prediction > 5:
        st.info("🛍 High-value customer → Target with premium offers.")
    elif prediction > 2:
        st.info("🛒 Mid-value customer → Use moderate promotions.")
    else:
        st.info("📌 Low-value customer → Awareness & bundle offers.")

# Footer
st.markdown("---")
st.markdown("**Developed by Prashant Singh (IIM Sirmaur)**  \nModel Training ▪ Clustering ▪ Deployment")
