import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import base64
from dotenv import load_dotenv
from email_utils import generate_pdf, send_email 

load_dotenv()

st.set_page_config(page_title="LungScan AI | Clinical Suite", page_icon="🫁", layout="wide")

# ---------------- PREDICTION LOGIC ----------------
@st.cache_resource
def load_models():
    image_model = load_model("Lung_Tumor.h5", compile=False)
    clinical_model = joblib.load("clinical_xgb_model.pkl")
    scaler = joblib.load("clinical_scaler.pkl")
    return image_model, clinical_model, scaler

image_model, clinical_model, scaler = load_models()

CLASS_NAMES = ["Benign", "Malignant", "Normal"]
IMG_SIZE = 160

def predict_image(img):
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)
    probs = image_model.predict(img_expanded)[0]
    return probs

def predict_clinical(features):
    features_scaled = scaler.transform([features])
    return clinical_model.predict_proba(features_scaled)[0][1]

# ---------------- ENHANCED UI & GLOBAL FONT BOOST ----------------
def set_ui_style():
    bg_path = "assets/bg_medical.jpeg"
    encoded_bg = ""
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            encoded_bg = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>
        /* GLOBAL FONT SIZE INCREASE */
        html, body, [class*="st-"] {{
            font-size: 18px !important; 
            color: #f1f5f9;
        }}
        
        .stApp {{
            background: linear-gradient(rgba(10, 15, 30, 0.92), rgba(10, 15, 30, 0.92)), 
                        url("data:image/jpeg;base64,{encoded_bg}") no-repeat center center fixed;
            background-size: cover;
        }}
        
        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background-color: rgba(7, 10, 20, 0.98) !important;
            border-right: 2px solid #38bdf8;
        }}

        .glass-panel {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(25px);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 25px;
            padding: 35px; margin-bottom: 25px;
        }}

        .main-header {{
            font-size: 3rem !important; font-weight: 900; text-align: center;
            background: linear-gradient(to right, #38bdf8, #818cf8, #38bdf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 50px;
        }}

        /* BUTTON */
        .stButton > button {{
            width: 100% !important; border-radius: 12px !important;
            height: 80px; font-size: 26px !important; font-weight: 900 !important;
            color: #ffffff !important; background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
            clip-path: polygon(5% 0%, 100% 0%, 95% 100%, 0% 100%);
            border: none !important; transition: 0.3s;
        }}
        
        /* REMOVE EMPTY BOX FEEL */
        [data-testid="stFileUploader"] {{
            background: rgba(15, 23, 42, 0.5);
            border: 2px dashed #38bdf8;
            border-radius: 15px;
            padding: 20px;
        }}

        .summary-card {{
            background: #ffffff !important; border-radius: 30px; padding: 50px;
            border-bottom: 10px solid #38bdf8;
        }}
        .summary-card * {{ color: #0f172a !important; font-size: 22px; }}
        
        .exec-summary-box {{
            background: #0f172a !important; border-radius: 20px; padding: 40px;
            margin-top: 30px; border: 1px solid #1e293b;
        }}
        .exec-summary-box p {{ font-size: 24px !important; color: #cbd5e1 !important; }}
        </style>
    """, unsafe_allow_html=True)

set_ui_style()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🫁</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;text-decoration: underline; margin-top: 0;'>LUNG TUMOR DETECTION FRAMEWORK</h3>", unsafe_allow_html=True)
    st.write("An AI-powered clinical decision support system combining medical imaging and clinical data.")
    st.markdown("---")
    st.title("⚙️ System Highlights")
    st.markdown("""
    - **🖥️ Imaging AI:** Deep CNN analyzes lung scans
    - **📊 Clinical Analysis:** XGBoost evaluates symptoms
    - **🔗 Multimodal Fusion:** Decision-level diagnosis
    - **📝 Smart Reports:** Hospital-grade PDF report
    - **📧 Secure Email:** Automated delivery
    """)
    st.divider()
    st.markdown("### 📊 System Intel")
    st.success("● Core Engine: Ready")
    st.success("● Multimodal Fusion: On")
    st.divider()
    st.warning("Clinical Decision Support Tool")
    st.warning("⚠️ **For clinical decision support only.** Not a replacement for medical professionals.")

# ---------------- MAIN UI ----------------
st.markdown('<h1 class="main-header">🫁 AI-based Multimodal Lung Tumor Detection</h1>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("👤 Patient Master Record")
    p_name = st.text_input("Full Patient Name")
    c1, c2 = st.columns(2)
    p_id = c1.text_input("🆔 Hospital ID")
    p_email = c2.text_input("✉️ Report Email")
    p_gender = c1.selectbox("⚧ Gender", ["Male", "Female"])
    p_age = c2.slider("📅 Age", 1, 100, 27)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader("🩺 Clinical Symptom Checklist")
    symptom_data = [
        ("Smoking", "🚬"), ("Yellow Fingers", "🖐️"), ("Anxiety", "🌀"),
        ("Peer Pressure", "👥"), ("Chronic Disease", "🏥"), ("Fatigue", "💤"),
        ("Allergic Rhinitis", "🤧"), ("Wheezing", "🌬️"), ("Alcohol", "🍷"),
        ("Coughing", "😷"), ("Shortness of Breath", "🫁"), 
        ("Swallowing Difficulty", "〰️"), ("Chest Pain", "⚠️")
    ]
    selected_values = []
    s_cols = st.columns(2)
    for i, (label, emoji) in enumerate(symptom_data):
        with (s_cols[0] if i % 2 == 0 else s_cols[1]):
            val = st.checkbox(f"{emoji} {label}")
            selected_values.append(2 if val else 1)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-panel" style="height:100%;">', unsafe_allow_html=True)
    st.subheader("⚕ Radiology CT Imaging")
    up_file = st.file_uploader("📥 Upload Radiology Scan (CT Scan)", type=["jpg", "png", "jpeg"])
    if up_file:
        img = Image.open(up_file).convert("RGB")
        st.markdown('<div class="scan-container">', unsafe_allow_html=True)
        st.image(img, use_container_width=True, caption="Radiology Scan Analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("⏳ Awaiting radiology data upload...")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FUSION & DIAGNOSIS ----------------
if st.button("🧠 GENERATE CLINICAL DIAGNOSIS"):
    if not up_file or not p_name:
        st.error("Please complete the profile and upload a scan.")
    else:
        with st.spinner("Processing Multimodal Vectors..."):
            # 1. Prediction Logic
            gender_enc = 1 if p_gender == "Male" else 0
            clinical_features = [gender_enc, p_age] + selected_values
            img_probs = predict_image(img)
            clin_prob = predict_clinical(clinical_features)
            
            # 2. Risk & Label Logic
            is_high_risk = clin_prob > 0.45 
            fusion_conf = int(img_probs[np.argmax(img_probs)] * 100)
            
            if img_probs[1] > 0.6 or (img_probs[1] > 0.35 and is_high_risk):
                diag_label, soft_hex, theme_bg, text_col = "Malignant", "#f87171", "rgba(248, 113, 113, 0.1)", "#fecaca"
                risk_score = "CRITICAL"
                reco = "IMMEDIATE ONCOLOGICAL REFERRAL: Priority PET-CT staging and core biopsy recommended."
                exec_summary = f"Patient {p_name} exhibits high-intensity radiological markers consistent with malignant proliferation. Immediate clinical correlation required."
            elif img_probs[0] > 0.5 or (img_probs[0] > 0.30 and is_high_risk):
                diag_label, soft_hex, theme_bg, text_col = "Benign", "#fbbf24", "rgba(251, 191, 36, 0.1)", "#fef3c7"
                risk_score = "MODERATE"
                reco = "CLINICAL MONITORING: 3-month follow-up thoracic imaging recommended."
                exec_summary = f"Radiological assessment for {p_name} shows a localized nodular mass with regular margins. Suggests benign etiology; monitoring required."
            else:
                diag_label, soft_hex, theme_bg, text_col = "Normal", "#34d399", "rgba(52, 211, 153, 0.1)", "#d1fae5"
                risk_score = "LOW"
                reco = "ROUTINE FOLLOW-UP: No acute pulmonary findings. Maintain standard screening."
                exec_summary = f"The diagnostic fusion for {p_name} reveals clear pulmonary architecture with no evidence of focal opacities."

            # 3. DARK THEME RENDER (Zero indentation on the HTML string)
            report_body = f"""
<div style="background: rgba(15, 23, 42, 0.8); color: white !important; border-radius: 20px; padding: 30px; border: 1px solid rgba(56, 189, 248, 0.3); border-top: 15px solid {soft_hex}; margin-top: 20px; font-family: sans-serif; backdrop-filter: blur(10px);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px;">
        <div>
            <h2 style="color: #38bdf8 !important; margin: 0; font-size: 1.6rem; font-weight: 800;">DIAGNOSTIC REPORT</h2>
            <p style="color: #94a3b8 !important; margin: 5px 0; font-size: 1rem;">
                <b>PATIENT:</b> {p_name} | <b>ID:</b> {p_id} | <b>AGE:</b> {p_age}
            </p>
        </div>
        <div style="text-align: right;">
            <div style="background: {theme_bg}; color: {soft_hex}; padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.8rem; display: inline-block;">{risk_score} RISK</div>
            <h1 style="color: {soft_hex} !important; margin: 5px 0 0 0; font-size: 2.8rem; font-weight: 900;">{diag_label.upper()}</h1>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
        <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
            <h4 style="color: #94a3b8 !important; margin: 0 0 15px 0; font-size: 0.8rem; text-transform: uppercase;">Probability Vectors</h4>
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1;"><span>MALIGNANT</span><span>{int(img_probs[1]*100)}%</span></div>
                <div style="background: #334155; height: 8px; border-radius: 4px;"><div style="width: {int(img_probs[1]*100)}%; background: #f87171; height: 100%; border-radius: 4px;"></div></div>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1;"><span>BENIGN</span><span>{int(img_probs[0]*100)}%</span></div>
                <div style="background: #334155; height: 8px; border-radius: 4px;"><div style="width: {int(img_probs[0]*100)}%; background: #fbbf24; height: 100%; border-radius: 4px;"></div></div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1;"><span>NORMAL</span><span>{int(img_probs[2]*100)}%</span></div>
                <div style="background: #334155; height: 8px; border-radius: 4px;"><div style="width: {int(img_probs[2]*100)}%; background: #34d399; height: 100%; border-radius: 4px;"></div></div>
            </div>
        </div>
        <div style="background: #0f172a; border: 1px solid #38bdf8; padding: 20px; border-radius: 15px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="color: #38bdf8; font-size: 0.8rem; font-weight: 800; text-transform: uppercase;">Confidence</div>
            <div style="color: white; font-size: 3rem; font-weight: 900;">{fusion_conf}<span style="font-size: 1.2rem;">%</span></div>
        </div>
    </div>
    <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; border-left: 6px solid #38bdf8; margin-bottom: 20px;">
        <h4 style="color: #38bdf8 !important; margin: 0 0 8px 0; font-size: 0.9rem;">CLINICAL SUMMARY</h4>
        <p style="color: #cbd5e1 !important; font-size: 1rem; line-height: 1.5; font-style: italic;">"{exec_summary}"</p>
    </div>
    <div style="background: {theme_bg}; border: 1px dashed {soft_hex}; border-radius: 10px; padding: 15px; display: flex; align-items: center; gap: 15px;">
        <div style="background: {soft_hex}; color: #0f172a; padding: 3px 10px; border-radius: 4px; font-weight: 900; font-size: 0.7rem;">ADVICE</div>
        <p style="color: {text_col} !important; font-size: 1.1rem; font-weight: 700; margin: 0;">{reco}</p>
    </div>
</div>
"""
            st.markdown(report_body, unsafe_allow_html=True)
            st.balloons()

            # --- 4. EMAIL & PDF AUTOMATION ---
            # --- 4. EMAIL & PDF AUTOMATION ---
            if p_email:
                try:
                    with st.spinner("📧 Delivering Secure PDF Report..."):
                        # 1. Generate PDF with all 9 required arguments
                        pdf_path = generate_pdf(
                            p_name, p_id, p_age, p_gender, 
                            diag_label, risk_score, exec_summary, reco, img
                        )
                        
                        # 2. Send to both Patient and Doctor
                        send_email(p_email, p_name, pdf_path)
                        
                        st.success(f"✅ Securely delivered to {p_email} and Clinical Team.")
                        
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                except Exception as e:
                    st.error(f"📬 Delivery Error: {str(e)}")
            else:
                st.info("ℹ️ No email provided. Report generated on-screen only.")