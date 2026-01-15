# app/app.py
# --------------------------------------------------
# Patient Survival Prediction App
# Streamlit Inference Application
# --------------------------------------------------


from pathlib import Path
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
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

# --------------------------------------------------
# Load Model & Feature Schema
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)
        return model, feature_names
    except Exception as e:
        st.error("🚨 Failed to load model artifacts.")
        st.stop()

model, feature_names = load_artifacts()
st.caption(f"Model loaded with {len(feature_names)} features")

# --------------------------------------------------
# Extract Valid Categories from Training Schema
# --------------------------------------------------

def extract_categories(prefix):
    return sorted(
        col.replace(prefix, "")
        for col in feature_names
        if col.startswith(prefix)
    )

drug_options = extract_categories("Treated_with_drugs_")
smoker_options = extract_categories("Patient_Smoker_")
location_options = extract_categories("Patient_Rural_Urban_")
mental_options = extract_categories("Patient_mental_condition_")

# --------------------------------------------------
# User Input Form
# --------------------------------------------------

st.subheader("📋 Patient Information")

with st.form("patient_form"):

    treated_with_drugs = st.selectbox("Treatment Type", drug_options)
    patient_age = st.slider("Patient Age", 1, 100, 45)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)

    smoker = st.selectbox("Smoker", smoker_options)
    rural_urban = st.selectbox("Location", location_options)
    mental_condition = st.selectbox("Mental Condition", mental_options)

    num_prev_conditions = st.number_input(
        "Number of Previous Conditions", 0, 10, 1
    )

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

    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("🧠 Prediction Result")

    if 0.4 < probability < 0.6:
        st.warning(
            "⚠️ Model confidence is moderate. "
            "Prediction should be interpreted cautiously."
        )

    if prediction == 1:
        st.success(
            f"✅ **Predicted Outcome: Survived**\n\n"
            f"**Probability:** {probability:.2%}"
        )
    else:
        st.error(
            f"❌ **Predicted Outcome: Not Survived**\n\n"
            f"**Probability:** {(1 - probability):.2%}"
        )
    # --------------------------------------------------
    # Feature Importance (Model Explainability)
    # --------------------------------------------------

    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(10))



    st.caption(
        "Top features influencing the model’s prediction "
        "(based on Gradient Boosting feature importance)."
      )
