import streamlit as st
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("Heart Disease Prediction")

if st.button("Predict"): 
    data = [[ 
        age,
        sex_map[sex],
        chest_pain_map[chest_pain],
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg_map[resting_ecg],
        max_hr,
        exercise_angina_map[exercise_angina],
        oldpeak,
        st_slope_map[st_slope]
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")