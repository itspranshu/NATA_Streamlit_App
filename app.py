import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ================================
# FILE PATHS (Your Desktop)
# ================================
MODEL_FILE = "nata_model.pkl"
FEATURE_FILE = "nata_features.json"

# ================================
# Streamlit Page Setup
# ================================
st.set_page_config(page_title="🛒 NATA Supermarket – Customer Insights App", layout="centered")
st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")

# ================================
# Load Model
# ================================
model = None
feature_cols = None

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found at Desktop.")
else:
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully from Desktop.")

if not os.path.exists(FEATURE_FILE):
    st.error("❌ Feature list not found at Desktop.")
else:
    with open(FEATURE_FILE, "r") as f:
        feature_cols = json.load(f)
        st.success("📄 Feature list loaded successfully.")

# ================================
# Input Section
# ================================
st.header("📥 Enter Customer Demographics & Behaviour")

input_data = {}

for feature in feature_cols:
    if feature in ["Education", "Marital_Status"]:
        input_data[feature] = st.selectbox(
            f"{feature} (Encoded Category)",
            ["0", "1", "2", "3", "4", "5", "6"]
        )
    else:
        input_data[feature] = st.number_input(feature, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("🔍 Input Preview")
st.dataframe(input_df)

# ================================
# Prediction
# ================================
if st.button("🔮 Predict Customer Spending"):
    if model is None:
        st.error("❌ Model not loaded.")
    else:
        X_pred = input_df[feature_cols]
        prediction = model.predict(X_pred)[0]

        st.success(f"💰 **Predicted Total Spending: ₹{prediction:,.2f}**")

        # Basic recommendation logic
        if prediction > 5:
            st.info("🛍 **High-value customer** → Recommend premium upselling, loyalty programs.")
        elif prediction > 2:
            st.info("🛒 **Mid-value customer** → Recommend moderate discounts, combo offers.")
        else:
            st.info("📌 **Low-value customer** → Awareness campaigns, entry-level offers.")


# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "**Developed by Prashant Singh (IIM Sirmaur)**  \n"
    "Model Training ▪ Clustering Analysis ▪ Hyperparameter Tuning ▪ Streamlit Deployment"
)



