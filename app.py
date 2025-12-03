%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# Ensure 'salary_model.pkl' is in the same directory/repo as this file
model = joblib.load('salary_model.pkl')

# 2. App Title & Description
st.title("Salary Predictor (Task 3)")
st.write("""
### Estimate Your Salary
Use the slider below to input years of experience. The model uses a **Decision Tree Regressor** trained on the Kaggle Salary dataset to predict the expected salary.
""")

# 3. Input Slider
years_exp = st.slider("Years of Experience", 0.0, 15.0, 5.0, step=0.1)

# 4. Prediction Button
if st.button("Predict Salary"):
    # Reshape input to 2D array [1 row, 1 column]
    input_data = np.array([[years_exp]])
    
    # Predict
    prediction = model.predict(input_data)
    
    # Display Result
    st.success(f"Estimated Salary: **${prediction[0]:,.2f}**")
