import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title('🎓 Student Score Prediction App')

# Collect user inputs
hours_studied = st.slider('📚 Hours Studied', min_value=0.0, max_value=24.0, step=0.5)
sleep_hours = st.slider('😴 Sleep Hours', min_value=0.0, max_value=24.0, step=0.5)
attendance_percent = st.slider('📅 Attendance Percent', min_value=0.0, max_value=100.0, step=1.0)
previous_scores = st.slider('🧮 Previous Scores', min_value=0, max_value=100, step=1)

# Predict button
if st.button('🔮 Predict'):
    # Prepare the input features as a 2D array
    input_data = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])

    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)

    # Make prediction using the model
    predicted_score = model.predict(scaled_data)

    # Display the prediction
    st.success(f'Score 🔰: {predicted_score[0]:.2f}')
