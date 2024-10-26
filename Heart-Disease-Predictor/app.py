import streamlit as st
import pandas as pd
import numpy as np
from joblib import load  # Or use pickle if preferred

# Load the trained model
model = load('Coronary-Heart-Disease-Risk-Prediction/Heart-Disease-Predictor/heart_disease_model.pkl')
scaler = load('Coronary-Heart-Disease-Risk-Prediction/Heart-Disease-Predictor/StandardScaler_HDModel.pkl')

# Define the app
st.title("Coronary Heart Disease Risk Prediction")

# Input fields for each feature
male = st.selectbox("Male (0 for female, 1 for male):", [0, 1])
age = st.number_input("Age:", min_value=1, max_value=120, value=50)
education = st.selectbox("Education Level (1-4):", [1, 2, 3, 4])
current_smoker = st.selectbox("Current Smoker (0 for No, 1 for Yes):", [0, 1])
if current_smoker == 1:
    cigs_per_day = st.number_input("Cigarettes per Day:", min_value=0, max_value=100, value=10)
else:
    cigs_per_day = 0  # Automatically set to 0 if not a smoker
    st.text("Cigarettes per Day: 0 (Not a smoker)")

bp_meds = st.selectbox("On Blood Pressure Medication (0 for No, 1 for Yes):", [0, 1])
prevalent_stroke = st.selectbox("History of Stroke (0 for No, 1 for Yes):", [0, 1])
prevalent_hyp = st.selectbox("Hypertension (0 for No, 1 for Yes):", [0, 1])
diabetes = st.selectbox("Diabetes (0 for No, 1 for Yes):", [0, 1])
tot_chol = st.number_input("Total Cholesterol:", min_value=100, max_value=600, value=200)
sys_bp = st.number_input("Systolic Blood Pressure:", min_value=80, max_value=250, value=120)
dia_bp = st.number_input("Diastolic Blood Pressure:", min_value=50, max_value=150, value=80)
bmi = st.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0)
heart_rate = st.number_input("Heart Rate:", min_value=40, max_value=200, value=70)
glucose = st.number_input("Glucose Level:", min_value=50, max_value=300, value=100)

# Collect input values in a DataFrame
input_data = pd.DataFrame({
    'male': [male],
    'age': [age],
    'education': [education],
    'currentSmoker': [current_smoker],
    'cigsPerDay': [cigs_per_day],
    'BPMeds': [bp_meds],
    'prevalentStroke': [prevalent_stroke],
    'prevalentHyp': [prevalent_hyp],
    'diabetes': [diabetes],
    'totChol': [tot_chol],
    'sysBP': [sys_bp],
    'diaBP': [dia_bp],
    'BMI': [bmi],
    'heartRate': [heart_rate],
    'glucose': [glucose]
})
feature_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']
input_data_tr = scaler.transform(input_data)
input_data_tr = pd.DataFrame(input_data_tr,columns=feature_names)
# Predict button
if st.button("Predict"):
    # Make prediction using the model
    prediction = model.predict(input_data_tr)[0]  # Assuming binary output (0 or 1)

    # Display result
    if prediction == 1:
        st.write("The model predicts a higher risk of coronary heart disease within 10 years.")
    else:
        st.write("The model predicts a lower risk of coronary heart disease within 10 years.")
