import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    # Example dataset (replace with a bigger dataset from Kaggle if possible)
    data = pd.DataFrame({
        "fever": [1, 0, 1, 0],
        "cough": [1, 1, 0, 0],
        "headache": [0, 1, 1, 1],
        "fatigue": [1, 0, 0, 1],
        "nausea": [0, 0, 1, 0],
        "anything":[1,0,1,0],
        "disease": ["Flu", "Cold", "Dengue", "Migraine"]
    })
    return data

data = load_data()

# -------------------------
# Train Model
# -------------------------
X = data.drop("disease", axis=1)
y = data["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🩺 Disease Prediction from Symptoms")
st.write("Select your symptoms and predict possible disease.")
x = st.slider("Select a value")
st.write(x, "squared is", x * x)
# Symptom checkboxes
fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")
nausea = st.checkbox("Nausea")
anything=st.checkbox("anything")
# Convert input to model format
user_input = [[
    1 if fever else 0,
    1 if cough else 0,
    1 if headache else 0,
    1 if fatigue else 0,
    1 if nausea else 0,
    1 if anything else 0
]]

# Predict button
if st.button("Predict Disease"):
    prediction = model.predict(user_input)[0]
    st.success(f"### 🧾 Predicted Disease: {prediction}")
