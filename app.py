import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px; padding: 36px 40px 32px;
    margin-bottom: 28px; border: 1px solid rgba(255,255,255,0.08);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(220,38,38,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-icon { font-size: 42px; margin-bottom: 10px; display: block; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 32px; color: #ffffff; margin: 0 0 8px; }
.hero-subtitle { color: rgba(255,255,255,0.55); font-size: 14px; font-weight: 400; margin: 0; }
.hero-badge {
    display: inline-block; margin-top: 14px;
    background: rgba(220,38,38,0.15); border: 1px solid rgba(220,38,38,0.35);
    color: #fca5a5; font-size: 11.5px; font-weight: 500;
    padding: 4px 12px; border-radius: 999px; letter-spacing: 0.5px;
}
.section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #94a3b8;
    margin: 24px 0 12px; display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #334155, transparent);
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 14px 0 !important;
    font-size: 15px !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(220,38,38,0.3) !important; margin-top: 8px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(220,38,38,0.4) !important;
}
.result-positive {
    background: linear-gradient(135deg, #1a1a2e, #2d1515);
    border: 1.5px solid rgba(220,38,38,0.5);
    border-radius: 16px; padding: 28px 32px; text-align: center; margin-top: 20px;
}
.result-negative {
    background: linear-gradient(135deg, #0a1628, #0d2818);
    border: 1.5px solid rgba(34,197,94,0.4);
    border-radius: 16px; padding: 28px 32px; text-align: center; margin-top: 20px;
}
.result-emoji { font-size: 48px; display: block; margin-bottom: 10px; }
.result-title-pos { font-family: 'DM Serif Display', serif; font-size: 24px; color: #fca5a5; margin: 0 0 6px; }
.result-title-neg { font-family: 'DM Serif Display', serif; font-size: 24px; color: #86efac; margin: 0 0 6px; }
.result-sub { font-size: 13.5px; color: rgba(255,255,255,0.5); margin: 0; line-height: 1.6; }
.result-disclaimer { font-size: 11px; color: rgba(255,255,255,0.3); margin-top: 14px; font-style: italic; }
.metric-row { display: flex; gap: 10px; margin-top: 14px; justify-content: center; }
.metric-chip {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 8px 14px; font-size: 12px;
    color: rgba(255,255,255,0.6); text-align: center;
}
.metric-chip b { display: block; font-size: 15px; color: #e2e8f0; font-weight: 600; }
.footer {
    text-align: center; margin-top: 36px; font-size: 12px; color: #94a3b8;
    padding-top: 20px; border-top: 1px solid #f1f5f9;
}
.footer a { color: #dc2626; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── Prediction Pipeline ───────────────────────────────────────────────────────
class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def predict(self, data):
        return self.model.predict(data)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon"></span>
    <p class="hero-title">Heart Disease Classifier</p>
    <p class="hero-subtitle">Clinical feature analysis powered by Logistic Regression · MLOps end-to-end pipeline</p>
    <span class="hero-badge">⚕ MLflow · DVC · DAGsHub Tracked</span>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1 — Demographics ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Demographics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=20, max_value=100, value=52, step=1)
with col2:
    Sex = st.selectbox("Sex", options=[0, 1],
                       format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")


# ── SECTION 2 — Cardiac Indicators ───────────────────────────────────────────
st.markdown('<div class="section-label">Cardiac Indicators</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    # schema col: "Chest pain type"
    Chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 — Typical Angina",
            2: "2 — Atypical Angina",
            3: "3 — Non-Anginal Pain",
            4: "4 — Asymptomatic"
        }[x]
    )
with col4:
    # schema col: "BP"
    BP = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=125)

col5, col6 = st.columns(2)
with col5:
    # schema col: "Cholesterol"
    Cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=213)
with col6:
    # schema col: "FBS over 120"
    FBS_over_120 = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

# schema col: "EKG results"
EKG_results = st.selectbox(
    "EKG Results",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "0 — Normal",
        1: "1 — ST-T Wave Abnormality",
        2: "2 — Left Ventricular Hypertrophy"
    }[x]
)


# ── SECTION 3 — Exercise & ST ─────────────────────────────────────────────────
st.markdown('<div class="section-label">Exercise & ST Analysis</div>', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    # schema col: "Max HR"
    Max_HR = st.slider("Max Heart Rate Achieved", min_value=70, max_value=210, value=168)
with col8:
    # schema col: "Exercise angina"
    Exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

col9, col10 = st.columns(2)
with col9:
    # schema col: "ST depression"  (float64)
    ST_depression = st.number_input(
        "ST Depression", min_value=0.0, max_value=7.0,
        value=2.0, step=0.1, format="%.1f"
    )
with col10:
    # schema col: "Slope of ST"
    Slope_of_ST = st.selectbox(
        "Slope of ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Upsloping",
            2: "2 — Flat",
            3: "3 — Downsloping"
        }[x]
    )


# ── SECTION 4 — Vessels & Thallium ───────────────────────────────────────────
st.markdown('<div class="section-label">Vessel & Thallium</div>', unsafe_allow_html=True)

col11, col12 = st.columns(2)
with col11:
    # schema col: "Number of vessels fluro"
    Number_of_vessels_fluro = st.selectbox(
        "Number of Major Vessels (Fluoroscopy)",
        options=[0, 1, 2, 3]
    )
with col12:
    # schema col: "Thallium"
    Thallium = st.selectbox(
        "Thallium Stress Test",
        options=[3, 6, 7],
        format_func=lambda x: {
            3: "3 — Normal",
            6: "6 — Fixed Defect",
            7: "7 — Reversible Defect"
        }[x]
    )


# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button(" Run Prediction")

if predict_btn:

    # Build DataFrame with EXACT column names from schema.yaml
    # 'id' excluded (not a feature), 'Heart Disease' excluded (target)
    input_data = pd.DataFrame([[
        Age, Sex, Chest_pain_type, BP, Cholesterol,
        FBS_over_120, EKG_results, Max_HR, Exercise_angina,
        ST_depression, Slope_of_ST, Number_of_vessels_fluro, Thallium
    ]], columns=[
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ])

    try:
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_data)[0]

        # Target is object type → handle both "Presence"/"Absence" and 1/0
        is_positive = (prediction == 1) or (str(prediction).strip().lower() in ["presence", "1", "yes"])

        if is_positive:
            st.markdown(f"""
            <div class="result-positive">
                <span class="result-emoji">⚠️</span>
                <p class="result-title-pos">High Risk Detected</p>
                <p class="result-sub">
                    The model predicts a <b style="color:#fca5a5;">positive indication</b>
                    of heart disease based on the provided clinical features.<br>
                    Immediate consultation with a cardiologist is strongly recommended.
                </p>
                <div class="metric-row">
                    <div class="metric-chip"><b>{prediction}</b>Raw Output</div>
                    <div class="metric-chip"><b>Presence</b>Classification</div>
                    <div class="metric-chip"><b>LogReg</b>Model</div>
                </div>
                <p class="result-disclaimer">⚕ For demo purposes only — not a medical diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <span class="result-emoji">✅</span>
                <p class="result-title-neg">Low Risk — Healthy Indicators</p>
                <p class="result-sub">
                    The model predicts <b style="color:#86efac;">no significant indication</b>
                    of heart disease based on the provided clinical features.<br>
                    Continue maintaining a heart-healthy lifestyle.
                </p>
                <div class="metric-row">
                    <div class="metric-chip"><b>{prediction}</b>Raw Output</div>
                    <div class="metric-chip"><b>Absence</b>Classification</div>
                    <div class="metric-chip"><b>LogReg</b>Model</div>
                </div>
                <p class="result-disclaimer">⚕ For demo purposes only — not a medical diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("⚠️ Model not found at `artifacts/model_trainer/model.joblib` — run `dvc repro` first.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.code(str(e))
        st.info("💡 Verify column names in your training data exactly match schema.yaml")


# ── Schema Reference Expander ─────────────────────────────────────────────────
with st.expander("📋 Schema Reference — Exact Column Names (schema.yaml)"):
    st.markdown("""
    | Schema Column | UI Label | Type | Values |
    |---------------|----------|------|--------|
    | `Age` | Age | int64 | 20–100 |
    | `Sex` | Sex | int64 | 0=Female, 1=Male |
    | `Chest pain type` | Chest Pain Type | int64 | 1, 2, 3, 4 |
    | `BP` | Blood Pressure | int64 | mmHg |
    | `Cholesterol` | Cholesterol | int64 | mg/dL |
    | `FBS over 120` | Fasting Blood Sugar | int64 | 0=No, 1=Yes |
    | `EKG results` | EKG Results | int64 | 0, 1, 2 |
    | `Max HR` | Max Heart Rate | int64 | 70–210 |
    | `Exercise angina` | Exercise Angina | int64 | 0=No, 1=Yes |
    | `ST depression` | ST Depression | **float64** | 0.0–7.0 |
    | `Slope of ST` | Slope of ST | int64 | 1, 2, 3 |
    | `Number of vessels fluro` | Vessels (Fluoroscopy) | int64 | 0–3 |
    | `Thallium` | Thallium Test | int64 | 3, 6, 7 |
    | ~~`id`~~ | *(excluded — index only)* | int64 | — |
    | ~~`Heart Disease`~~ | *(target — not an input)* | object | Presence/Absence |
    """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <b>Asadullah Shehbaz</b> · Kaggle Datasets Grandmaster 🏅 ·
    <a href="https://www.kaggle.com/asadullahcreative" target="_blank">Kaggle</a> ·
    <a href="https://github.com/AsadullahShehbaz" target="_blank">GitHub</a> ·
    <a href="https://www.fiverr.com/asadullah_92" target="_blank">Fiverr</a>
</div>
""", unsafe_allow_html=True)