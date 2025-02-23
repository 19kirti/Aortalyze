import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart.csv")

# Identify categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
label_encoders = {}

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for later use

# Define features and target
X = df.drop(columns=["HeartDisease"])  # Features
y = df["HeartDisease"]  # Target

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model and save it using pickle
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the model and encoders
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Heart Disease Prediction System")

# User inputs (Main Page, Not Sidebar)
sex = st.selectbox("Sex", label_encoders["Sex"].classes_)
chest_pain = st.selectbox("Chest Pain Type", label_encoders["ChestPainType"].classes_)
resting_ecg = st.selectbox("Resting ECG", label_encoders["RestingECG"].classes_)
exercise_angina = st.selectbox("Exercise Angina", label_encoders["ExerciseAngina"].classes_)
st_slope = st.selectbox("ST Slope", label_encoders["ST_Slope"].classes_)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
resting_bp = st.number_input("Resting BP", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.5)

# Convert categorical inputs using label encoders
encoded_inputs = []
for col, value in zip(categorical_cols, [sex, chest_pain, resting_ecg, exercise_angina, st_slope]):
    if value in label_encoders[col].classes_:
        encoded_inputs.append(label_encoders[col].transform([value])[0])
    else:
        encoded_inputs.append(-1)  # Assign a default value for unseen labels

# Create a list of user inputs
user_inputs = encoded_inputs + [age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak]

# Convert list to DataFrame
user_inputs_df = pd.DataFrame([user_inputs], columns=X.columns)

# Standardize user input
user_scaled = scaler.transform(user_inputs_df)

# Prediction
if st.button("Predict"):
    prediction = rf_model.predict(user_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High risk of heart disease! Consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease. Stay healthy!")

st.write("User Inputs:", user_inputs)
st.write("Encoded Inputs:", encoded_inputs)  # Check if categorical values are properly transformed
st.write("Scaled Inputs:", user_scaled)  # Verify that scaling is correct
