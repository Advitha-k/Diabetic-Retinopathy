import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetic Retinopathy Screening",
    layout="wide"
)

# ---------------- CSS RESET + NEW DESIGN ----------------
st.markdown(
    """
    <style>

    /* ===== FULL PAGE RESET ===== */
    html, body, [class*="css"]  {
        font-family: "Segoe UI", sans-serif;
        background-color: #EAF4FF;
    }

    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #B7D3EE;
    }

    /* Remove top white bar */
    [data-testid="stHeader"] {
        background-color: #B7D3EE;
    }

    /* Content container */
    .block-container {
        padding: 2.5rem 3rem;
    }

    /* ===== LEFT SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #9ABFE3;
        padding: 2rem 1.5rem;
    }
    [data-testid="stSidebar"] h3 {
    color: #0B5394;
}

    /* Sidebar text */
    [data-testid="stSidebar"] h2 {
        color: #0B5394;
    }

    [data-testid="stSidebar"] p {
        color: #0B5394;
        font-size: 15px;
    }

    /* ===== TITLES ===== */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #0B5394;
        margin-bottom: 5px;
    }

    [data-testid="stFileUploader"] label {
    color: #0B5394 !important;
}


    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    /* ===== RESULT CARD ===== */
    .result-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        margin-top: 25px;
    }

    .label {
        font-weight: 600;
        color: #0B5394;
        margin-top: 10px;
    }

    /* Upload box */
    section[data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR CONTENT ----------------
st.sidebar.markdown("##  About")
st.sidebar.markdown(
    """
    This AI-based system analyzes **retinal fundus images** to
    detect **Diabetic Retinopathy stages**.

    ### What it provides:
    • DR Stage classification  
    • Confidence score  
    • Health risk level  
    • Medical guidance  

    ### Important:
    This tool is **educational** and does **not replace** a medical diagnosis.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(" *Eye health saves vision.*")

# ---------------- MAIN CONTENT ----------------
st.markdown('<div class="main-title">Diabetic Retinopathy Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based retinal image analysis for early eye health assessment</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("dr_model.h5")

DR_CLASSES = {
    0: "No Diabetic Retinopathy",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

RISK_MAPPING = {
    0: ("Very Low Risk", "No immediate consultation needed"),
    1: ("Low Risk", "Routine eye check recommended"),
    2: ("Medium Risk", "Consult an ophthalmologist"),
    3: ("High Risk", "Urgent medical consultation advised"),
    4: ("Critical Risk", "Immediate medical attention required")
}

SUMMARY_MAPPING = {
    0: "The retina appears healthy with no visible diabetic damage.",
    1: "Early-stage changes detected. Monitoring can prevent progression.",
    2: "Moderate retinal damage observed. Medical review is important.",
    3: "Severe damage detected. Prompt treatment is strongly advised.",
    4: "Advanced retinal damage detected. Emergency care is required."
}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Retinal Image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", width=350)

    if st.button("🔍 Analyze Image"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100
        risk, advice = RISK_MAPPING[class_index]
        summary = SUMMARY_MAPPING[class_index]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(f"<div class='label'>DR Stage</div>{DR_CLASSES[class_index]}", unsafe_allow_html=True)
        st.markdown(f"<div class='label'>Confidence</div>{confidence:.2f}%", unsafe_allow_html=True)
        st.markdown(f"<div class='label'>Health Risk Level</div>{risk}", unsafe_allow_html=True)
        st.markdown(f"<div class='label'>Medical Advice</div>{advice}", unsafe_allow_html=True)

        st.markdown("<div class='label'>Summary</div>", unsafe_allow_html=True)
        st.info(summary)

        st.warning(
            "⚠️ This application is for educational purposes only and should not be used as a medical diagnosis."
        )

        st.markdown('</div>', unsafe_allow_html=True)
