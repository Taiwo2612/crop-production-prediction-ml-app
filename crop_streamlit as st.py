import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load model and encoders
# ----------------------------
model = joblib.load("crop_model.pkl")

state_encoder = joblib.load("state_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")

feature_order = joblib.load("model_features.pkl")

# ----------------------------
# App title
# ----------------------------
st.title("🌾 Crop Production Prediction System")

st.write("Enter the details below to predict crop production.")

# ----------------------------
# User Inputs
# ----------------------------

state = st.selectbox(
    "Select State",
    state_encoder.classes_
)

crop = st.selectbox(
    "Select Crop",
    crop_encoder.classes_
)

season = st.selectbox(
    "Select Season",
    season_encoder.classes_
)

crop_year = st.number_input("Crop Year", min_value=1990, max_value=2035)

area = st.number_input("Area (hectares)", min_value=0.0)

rainfall = st.number_input("Annual Rainfall")

fertilizer = st.number_input("Fertilizer Usage")

pesticide = st.number_input("Pesticide Usage")

# ----------------------------
# Encode Inputs
# ----------------------------

state_encoded = state_encoder.transform([state])[0]
crop_encoded = crop_encoder.transform([crop])[0]
season_encoded = season_encoder.transform([season])[0]

# ----------------------------
# Create Input DataFrame
# ----------------------------

input_data = pd.DataFrame([[
    state_encoded,
    crop_encoded,
    season_encoded,
    crop_year,
    area,
    rainfall,
    fertilizer,
    pesticide
]], columns=feature_order)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Production"):

    prediction = model.predict(input_data)

    st.success(f"🌾 Predicted Crop Production: {prediction[0]:,.2f}")