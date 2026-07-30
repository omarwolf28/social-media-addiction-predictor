"""
Streamlit front-end for the Addiction Level model (Standalone Version).
No separate FastAPI server required.
"""

import os
import joblib
import streamlit as st

# Import preprocessing logic and options directly from preprocess.py[cite: 9]
from preprocess import (
    RELATIONSHIP_OPTIONS, 
    OCCUPATION_OPTIONS, 
    AVG_TIME_OPTIONS, 
    RATING_FIELDS,
    preprocess_input
)

# ----------------------------------------------------------------------
# 1. PAGE CONFIGURATION[cite: 11]
# ----------------------------------------------------------------------
st.set_page_config(page_title="Social Media Addiction Level", page_icon="📱")

st.title("📱 Social Media Addiction Level Predictor")
st.write(
    "Answer the questions below and get a prediction of your Addiction Level "
    "(Low, Moderate, High, or Severe), based on a model trained on survey data."
)

# ----------------------------------------------------------------------
# 2. LOAD ALL ML FILES
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    # Because this script runs in the 'deployment' folder, we point to the 'model' subfolder
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    
    model = joblib.load(os.path.join(MODEL_DIR, "model_addiction.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le_gender = joblib.load(os.path.join(MODEL_DIR, "le_gender.pkl"))
    encoder_avg_time = joblib.load(os.path.join(MODEL_DIR, "encoder_avg_time.pkl"))
    encoder_addiction = joblib.load(os.path.join(MODEL_DIR, "encoder_addiction.pkl"))
    
    return model, scaler, le_gender, encoder_avg_time, encoder_addiction

model, scaler, le_gender, encoder_avg_time, encoder_addiction = load_models()

# ----------------------------------------------------------------------
# 3. UI ELEMENTS[cite: 11]
# ----------------------------------------------------------------------
with st.form("survey_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=22)
        gender = st.selectbox("Gender", ["Female", "Male"])
        relationship_status = st.selectbox("Relationship Status", RELATIONSHIP_OPTIONS)
        occupation_status = st.selectbox("Occupation Status", OCCUPATION_OPTIONS)

    with col2:
        affiliate_organization = st.text_input(
            "Affiliated Organizations (comma separated, or N/A)", value="N/A"
        )
        platforms = st.text_input(
            "Social Media Platforms Used (comma separated)", value="Instagram, TikTok"
        )
        avg_time_per_day = st.selectbox("Average Time Spent Per Day", AVG_TIME_OPTIONS)

    st.subheader("Rate the following from 1 (never/not at all) to 5 (always/extremely)")

    rating_labels = {
        'use_without_purpose(1-5)': "Using social media without a specific purpose",
        'distracted(1-5)': "Getting distracted by social media while busy",
        'restless(1-5)': "Feeling restless without social media",
        'distracted_easily(1-5)': "How easily distracted in general",
        'worries(1-5)': "Being bothered by worries",
        'concentration(1-5)': "Difficulty concentrating",
        'compare_to_others(1-5)': "Comparing yourself to others on social media",
        'compare_feelings(1-5)': "How you feel about those comparisons",
        'validation(1-5)': "Seeking validation from social media",
        'depressed(1-5)': "Feeling depressed or down",
        'daily_activity_fluctuate(1-5)': "Interest in daily activities fluctuating",
        'sleeping_issues(1-5)': "Facing sleep issues",
    }

    ratings = {}
    for field in RATING_FIELDS:
        ratings[field] = st.slider(rating_labels[field], 1, 5, 3)

    submitted = st.form_submit_button("Predict Addiction Level")

# ----------------------------------------------------------------------
# 4. DIRECT PREDICTION LOGIC[cite: 8]
# ----------------------------------------------------------------------
if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "relationship_status": relationship_status,
        "occupation_status": occupation_status,
        "affiliate_organization": affiliate_organization,
        "platforms": platforms,
        "avg_time_per_day": avg_time_per_day,
        **ratings,
    }

    try:
        # 1. Clean and format the data using your shared preprocess.py function[cite: 9]
        X_ready = preprocess_input(payload, le_gender, encoder_avg_time, scaler)

        # 2. Make the prediction directly[cite: 8]
        pred_idx = model.predict(X_ready)
        
        # 3. Decode the prediction back into the text label[cite: 8]
        pred_label = encoder_addiction.inverse_transform(pred_idx.reshape(-1, 1))[0][0]

        # 4. Check for confidence score if the model supports predict_proba[cite: 8]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(X_ready)[0]))

        # 5. Display results[cite: 11]
        st.success(f"Predicted Addiction Level: **{pred_label}**")
        if confidence > 0:
            st.caption(f"Model confidence: {confidence * 100:.1f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")