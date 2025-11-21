import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ================================
# File Paths
# ================================
MODEL_FILE = "nata_model.pkl"
FEATURE_FILE = "nata_features.json"
DEFAULTS_FILE = "default_values.json"

# ================================
# Streamlit Page Setup
# ================================
st.set_page_config(page_title="🛒 NATA Supermarket – Customer Insights", layout="centered")
st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")

# ================================
# Load Model and Files
# ================================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file missing.")

if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, "r") as f:
        feature_cols = json.load(f)
else:
    st.error("❌ Features file missing.")

if os.path.exists(DEFAULTS_FILE):
    with open(DEFAULTS_FILE, "r") as f:
        default_values = json.load(f)
else:
    st.error("❌ Default values file missing.")

# ================================
# Select 7–8 Features for UI
# ================================
user_input_features = [
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Education",
    "Marital_Status"
]

st.header("📥 Enter Customer Details")

input_data = {}
for feature in user_input_features:
    if feature in ["Education", "Marital_Status"]:
        input_data[feature] = st.selectbox(
            feature, ["0", "1", "2", "3", "4", "5", "6"]
        )
    else:
        input_data[feature] = st.number_input(feature, value=0.0)

# ================================
# Build Final Input Vector
# ================================
final_input = {}

for feature in feature_cols:
    if feature in user_input_features:
        final_input[feature] = input_data[feature]
    else:
        final_input[feature] = default_values.get(feature)

input_df = pd.DataFrame([final_input])

st.subheader("🔍 Input Preview")
st.dataframe(input_df)

# ================================
# Predict
# ================================
if st.button("🔮 Predict Spending"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Total Spending: **₹{prediction:,.2f}**")

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown(
    "**Developed by Prashant Singh (IIM Sirmaur)**  \n"
    "Model Training ▪ Insights Engine ▪ Streamlit Deployment"
)
