# app/app.py
# --------------------------------------------------
# Patient Survival Prediction App
# Streamlit Inference Application
# --------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Patient Survival Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Patient Survival Prediction System")
st.write(
    """
    This application predicts whether a patient is likely to survive
    one year after treatment based on demographic, clinical, and treatment data.
    """
)

# --------------------------------------------------
# Load Model & Feature Schema
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/gradient_boosting.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

# --------------------------------------------------
# User Input Form
# --------------------------------------------------

st.subheader("📋 Patient Information")

with st.form("patient_form"):

    treated_with_drugs = st.selectbox(
        "Treatment Type",
        [
            "DX1", "DX2", "DX3", "DX4",
            "DX1 DX2", "DX1 DX3", "DX2 DX3",
            "DX1 DX2 DX3 DX4", "DX6"
        ]
    )

    patient_age = st.slider("Patient Age", 1, 100, 45)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)

    smoker = st.selectbox("Smoker", ["Yes", "No"])
    rural_urban = st.selectbox("Location", ["Urban", "Rural"])
    mental_condition = st.selectbox(
        "Mental Condition",
        ["Stable", "Unstable"]
    )

    num_prev_conditions = st.number_input(
        "Number of Previous Conditions",
        0, 10, 1
    )

    # Binary clinical indicators
    st.markdown("**Clinical Indicators**")
    col1, col2, col3 = st.columns(3)

    with col1:
        A = st.selectbox("A", [0, 1])
        B = st.selectbox("B", [0, 1])

    with col2:
        C = st.selectbox("C", [0, 1])
        D = st.selectbox("D", [0, 1])

    with col3:
        E = st.selectbox("E", [0, 1])
        F = st.selectbox("F", [0, 1])

    Z = st.selectbox("Z", [0, 1])

    submitted = st.form_submit_button("🔍 Predict Survival")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

if submitted:

    # Create input DataFrame (EXACT training columns)
    input_data = pd.DataFrame([{
        "Treated_with_drugs": treated_with_drugs,
        "Patient_Age": patient_age,
        "Patient_Body_Mass_Index": bmi,
        "Patient_Smoker": smoker,
        "Patient_Rural_Urban": rural_urban,
        "Patient_mental_condition": mental_condition,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "E": E,
        "F": F,
        "Z": Z,
        "Number_of_prev_cond": num_prev_conditions
    }])

    # One-hot encode
    input_encoded = pd.get_dummies(input_data)

    # Align with training features
    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("🧠 Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ **Predicted Outcome: Survived**\n\n"
            f"**Probability:** {probability:.2%}"
        )
    else:
        st.error(
            f"❌ **Predicted Outcome: Not Survived**\n\n"
            f"**Probability:** {1 - probability:.2%}"
        )

    st.caption(
        "⚠️ This prediction is for educational purposes only and "
        "should not be used for real medical decisions."
    )
