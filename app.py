import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn # Explicitly import sklearn to ensure pickle can resolve the model class

# 1. Page Configuration
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# 2. Load the Model
@st.cache_resource
def load_model():
    try:
        # Load the model using pickle (rb = read binary)
        with open('salary_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'salary_model.pkl' not found. Please ensure the model file is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

model = load_model()

# 3. UI Design
st.title("💰 Salary vs. Experience Predictor")
st.markdown("This app uses a **Decision Tree Regressor** to predict salary based on years of experience.")
st.markdown("---")

# 4. User Input
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910791.png", width=100)

with col2:
    years_experience = st.number_input(
        "Enter Years of Experience:",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1,
        help="Enter the total years of professional experience."
    )

# 5. Prediction Logic
if st.button("Predict Salary", type="primary"):
    if model is not None:
        # The model expects a 2D array.
        input_data = np.array([[years_experience]])
        
        try:
            prediction = model.predict(input_data)
            
            # Display Result
            st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
            st.info(f"The model estimates this salary based on {years_experience} years of experience.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("Model not loaded. Please check the file structure.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn | Deployed via GitHub")
