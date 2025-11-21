import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ================================
# FILE PATHS
# ================================
MODEL_FILE = "nata_model.pkl"
FEATURE_FILE = "nata_features.json"
DEFAULTS_FILE = "default_values.json"

st.set_page_config(page_title="NATA Customer Prediction App", layout="centered")

st.title("🛒 NATA Supermarket – Customer Insights & Spending Prediction")


# ================================
# Load Model + Files
# ================================
model, feature_cols, default_vals = None, None, None

# Load model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file NOT found.")

# Load features list
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, "r") as f:
        feature_cols = json.load(f)
else:
    st.error("❌ Feature list NOT found.")

# Load default values
if os.path.exists(DEFAULT_FILE):
    with open(DEFAULT_FILE, "r") as f:
        default_vals = json.load(f)
else:
    st.error("❌ Default values file NOT found.")

# ================================
# CHOOSE 7 USER INPUT VARIABLES
# ================================
USER_INPUT_VARS = [
    "Year_Birth",          # Customer birth year
    "Income",              # Annual income
    "Kidhome",             # Number of kids
    "Teenhome",            # Number of teenagers
    "Recency",             # Days since last purchase
    "Education",           # Encoded education category
    "Marital_Status"       # Encoded marital status
]

st.header("📥 Enter Customer Information")

input_data = {}

for feature in USER_INPUT_VARS:
    if feature in ["Education", "Marital_Status"]:
        input_data[feature] = st.selectbox(
            f"{feature} (Category Code)",
            ["0","1","2","3","4","5","6"],
            index=0
        )
    else:
        input_data[feature] = st.number_input(
            feature,
            value=float(default_vals.get(feature, 0.0))
        )

# ================================
# Construct full feature row
# ================================
full_data = {}

for col in feature_cols:
    if col in USER_INPUT_VARS:
        full_data[col] = input_data[col]
    else:
        # Fill missing features with default values
        full_data[col] = default_vals.get(col, 0)

# Convert to DataFrame
input_df = pd.DataFrame([full_data])

st.subheader("🔍 Final Input Used for Prediction")
st.dataframe(input_df)

# ================================
# Prediction
# ================================
if st.button("🔮 Predict Customer Spending"):
    if model is None:
        st.error("❌ Model not available.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 **Predicted Total Spending: ₹{prediction:,.2f}**")

        # Recommendations
        if prediction > 2500:
            st.info("🛍 High-value customer → Recommend premium loyalty offers.")
        elif prediction > 1000:
            st.info("🛒 Mid-value customer → Suggest combos & moderate discounts.")
        else:
            st.info("📌 Low-value customer → Awareness campaigns, entry-level offers.")

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown(
    "**Developed by Prashant Singh (IIM Sirmaur)**  \n"
    "Model Training ▪ Clustering Analysis ▪ Streamlit Deployment ▪ UI Design"
)

