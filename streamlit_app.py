import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model (make sure you saved it earlier using joblib.dump)
model = joblib.load("disease_model.pkl")

st.title("🧑‍⚕️ Disease Prediction App")

# --- Input Fields ---
st.header("Enter Patient Details:")

age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400)
symptom1 = st.selectbox("Cough", ["No", "Yes"])
symptom2 = st.selectbox("Fever", ["No", "Yes"])

# Convert categorical values to numeric (if your model needs it)
gender_num = 1 if gender == "Male" else 0
symptom1_num = 1 if symptom1 == "Yes" else 0
symptom2_num = 1 if symptom2 == "Yes" else 0

# Create input vector
user_data = np.array([[age, gender_num, blood_pressure, cholesterol, symptom1_num, symptom2_num]])

# --- Prediction Button ---
if st.button("Predict Disease"):
    prediction = model.predict(user_data)
    st.success(f"Predicted Disease: {prediction[0]}")
