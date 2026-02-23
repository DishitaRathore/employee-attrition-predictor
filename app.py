import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load('attrition_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first to generate the models.")
    st.stop()

# Building the Streamlit UI
st.title("Employee Attrition Predictor 🏢")
st.write("Enter the employee's details below to predict their likelihood of leaving the company.")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    monthly_income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=25000, value=5000, step=500)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)

with col2:
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_satisfaction = st.selectbox("Job Satisfaction Rating", [1, 2, 3, 4], index=2, help="1 = Low, 4 = Very High")
    over_time = st.selectbox("Works OverTime?", ["Yes", "No"])

if st.button("Predict Attrition Risk", type="primary"):
    
    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'TotalWorkingYears': [total_working_years],
        'Department': [department],
        'JobSatisfaction': [job_satisfaction],
        'OverTime': [over_time]
    })
    
    # One-hot encoding the inputs to match training data
    input_encoded = pd.get_dummies(input_data, columns=['Department', 'OverTime'])
    
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Reordering the columns to perfectly match the model's training structure
    input_encoded = input_encoded[model_columns]
    
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Attrition!**")
        st.write(f"The model predicts a **{probability:.1%}** probability that this employee will leave.")
    else:
        st.success(f"✅ **Low Risk of Attrition.**")
        st.write(f"The model predicts only a **{probability:.1%}** probability that this employee will leave.")