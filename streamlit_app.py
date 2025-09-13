import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Title
# ---------------------------
st.title("🩺 Disease Prediction App")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    # Example dataset: replace with your dataset
    data = {
        "fever": [98, 101, 100, 102, 97, 99],
        "cough": [1, 1, 0, 1, 0, 0],
        "fatigue": [1, 0, 1, 1, 0, 0],
        "disease": ["Flu", "Flu", "Healthy", "Flu", "Healthy", "Healthy"]
    }
    return pd.DataFrame(data)

df = load_data()

# ---------------------------
# Train Model
# ---------------------------
X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter Patient Symptoms:")

fever = st.number_input("Fever (°F)", min_value=95, max_value=110, value=98)
cough = st.selectbox("Cough", [0, 1])  # 0 = No, 1 = Yes
fatigue = st.selectbox("Fatigue", [0, 1])  # 0 = No, 1 = Yes

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([[fever, cough, fatigue]], columns=["fever", "cough", "fatigue"])
    prediction = model.predict(input_data)[0]
    st.success(f"🧾 Predicted Disease: **{prediction}**")

# ---------------------------
# Dataset Preview
# ---------------------------
with st.expander("See Training Data"):
    st.write(df)
