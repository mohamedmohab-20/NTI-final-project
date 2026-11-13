import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Kidney Health Assessment | KDIGO Risk Stratification",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 24px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .stage-normal {
        color: #1e3a8a;
        font-weight: 600;
        font-size: 18px;
    }
    .stage-micro {
        color: #b45309;
        font-weight: 600;
        font-size: 18px;
    }
    .stage-macro {
        color: #b91c1c;
        font-weight: 600;
        font-size: 18px;
    }
    .pipeline-step {
        background-color: #f1f5f9;
        border-left: 3px solid #0284c7;
        padding: 8px 12px;
        font-size: 13px;
        margin-bottom: 8px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    retained_cols = joblib.load("retained_features.pkl")
    proc_cols = joblib.load("processed_feature_names.pkl")
    ort_session = ort.InferenceSession("kidney_model.onnx")
    
    explainer = None
    if os.path.exists("shap_explainer.pkl"):
        explainer = joblib.load("shap_explainer.pkl")
    return preprocessor, retained_cols, proc_cols, ort_session, explainer

preprocessor, retained_cols, proc_cols, ort_session, explainer = load_artifacts()

st.markdown('<div class="main-title">Clinical Kidney Health Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">KDIGO 3-Class Albuminuria Risk Stratification | ONNX Runtime Engine</div>', unsafe_allow_html=True)

st.markdown("""
<div class="metric-box">
    <strong>Inference Pipeline Architecture</strong>
    <div style="display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap;">
        <div class="pipeline-step">1. Routine Clinical Biomarkers</div>
        <div class="pipeline-step">2. MICE Imputation & Standardization</div>
        <div class="pipeline-step">3. ONNX Neural Runtime Session</div>
        <div class="pipeline-step">4. KDIGO Staging & SHAP Attribution</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.form("assessment_form"):
    st.subheader("Patient Clinical & Laboratory Biomarkers")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", min_value=18, max_value=90, value=56)
        gender_str = st.selectbox("Gender", ["Male", "Female"])
        gender = 1.0 if gender_str == "Male" else 2.0
        ethnicity_dict = {
            "Non-Hispanic White": 3.0,
            "Non-Hispanic Black": 4.0,
            "Mexican American": 1.0,
            "Other Hispanic": 2.0,
            "Non-Hispanic Asian": 6.0,
            "Other / Multi-Racial": 7.0
        }
        ethnicity_str = st.selectbox("Race / Ethnicity", list(ethnicity_dict.keys()))
        ethnicity = ethnicity_dict[ethnicity_str]
        income_ratio = st.number_input("Poverty-Income Ratio (PIR)", min_value=0.0, max_value=5.0, value=2.4, step=0.1)

    with c2:
        st.markdown("**Anthropometrics & Vitals**")
        bmi = st.number_input("Body Mass Index (kg/m2)", min_value=12.0, max_value=65.0, value=28.4, step=0.1)
        heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40.0, max_value=140.0, value=74.0, step=1.0)
        alcohol = st.number_input("Alcohol Drinks / Week", min_value=0.0, max_value=60.0, value=2.0, step=1.0)
        smoking_dict = {"Never / Not at all": 3.0, "Some days": 2.0, "Every day": 1.0}
        smoking_str = st.selectbox("Smoking Status", list(smoking_dict.keys()))
        smoking = smoking_dict[smoking_str]

    with c3:
        st.markdown("**CBC & Renal Biomarkers**")
        creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.3, max_value=6.0, value=1.15, step=0.05)
        uric_acid = st.number_input("Serum Uric Acid (mg/dL)", min_value=1.5, max_value=15.0, value=5.6, step=0.1)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=6.0, max_value=20.0, value=14.1, step=0.1)
        wbc = st.number_input("White Blood Cells (1000/uL)", min_value=2.0, max_value=25.0, value=6.8, step=0.1)
        platelets = st.number_input("Platelets (1000/uL)", min_value=50.0, max_value=700.0, value=235.0, step=5.0)

    with c4:
        st.markdown("**Hepatic & Electrolyte Panel**")
        ggt = st.number_input("Liver GGT (U/L)", min_value=5.0, max_value=350.0, value=28.0, step=1.0)
        ast = st.number_input("Liver AST (U/L)", min_value=5.0, max_value=250.0, value=24.0, step=1.0)
        bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=5.0, value=0.7, step=0.1)
        mcv = st.number_input("Mean Corpuscular Volume (fL)", min_value=60.0, max_value=120.0, value=89.5, step=0.5)
        sodium = st.number_input("Serum Sodium (mmol/L)", min_value=125.0, max_value=155.0, value=139.0, step=0.5)
        potassium = st.number_input("Serum Potassium (mmol/L)", min_value=2.5, max_value=6.5, value=4.1, step=0.1)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=90.0, max_value=400.0, value=198.0, step=1.0)

    submitted = st.form_submit_button("Run Clinical Risk Assessment")

if submitted:
    raw_data = {
        'age': float(age),
        'gender': float(gender),
        'ethnicity': float(ethnicity),
        'income_ratio': float(income_ratio),
        'body_mass_index': float(bmi),
        'heart_rate_bpm': float(heart_rate),
        'white_blood_cells_count': float(wbc),
        'platelets_count': float(platelets),
        'hemoglobin_g_dl': float(hemoglobin),
        'mean_corpuscular_volume_fL': float(mcv),
        'creatinine_mg_dl': float(creatinine),
        'liver_ast_U_L': float(ast),
        'bilirubin_mg_dl': float(bilirubin),
        'liver_ggt_U_L': float(ggt),
        'uric_acid_mg_dl': float(uric_acid),
        'sodium_mmol_L': float(sodium),
        'potassium_mmol_L': float(potassium),
        'cholesterol_mg_dl': float(cholesterol),
        'alcohol_drinks_per_week': float(alcohol),
        'smoking_status': float(smoking)
    }
    
    input_df = pd.DataFrame([raw_data])[retained_cols]
    proc_arr = preprocessor.transform(input_df).astype(np.float32)
    proc_df = pd.DataFrame(proc_arr, columns=proc_cols)
    
    ort_inputs = {ort_session.get_inputs()[0].name: proc_arr}
    ort_out = ort_session.run(None, ort_inputs)[0][0]
    
    pred_stage = int(np.argmax(ort_out))
    stage_labels = ["Normal", "Microalbuminuria", "Macroalbuminuria"]
    stage_classes = ["stage-normal", "stage-micro", "stage-macro"]
    stage_criteria = [
        "ACR < 30 mg/g (KDIGO Stage A1: Low risk)",
        "30 <= ACR < 300 mg/g (KDIGO Stage A2: Moderately increased risk)",
        "ACR >= 300 mg/g (KDIGO Stage A3: Severely increased risk)"
    ]
    
    st.markdown("---")
    st.subheader("Risk Stratification Results")
    
    r1, r2 = st.columns([1, 1])
    
    with r1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size: 13px; color: #64748b;">Predicted KDIGO Risk Stage</div>
            <div class="{stage_classes[pred_stage]}">{stage_labels[pred_stage]}</div>
            <div style="font-size: 13px; color: #475569; margin-top: 6px;">{stage_criteria[pred_stage]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(6, 2.8))
        y_pos = np.arange(len(stage_labels))
        bars = ax.barh(y_pos, ort_out * 100, color=['#2563eb', '#d97706', '#dc2626'], height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stage_labels, fontsize=10)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2., f"{val:.1f}%",
                    va='center', ha='left', fontsize=9, color='#334155')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with r2:
        st.markdown("**Explainable AI: Local Biomarker Attribution**")
        if explainer is not None:
            try:
                shap_res = explainer(proc_df)
                shap_vals = shap_res.values[0]
                if len(shap_vals.shape) == 2:
                    target_shap = shap_vals[:, pred_stage]
                else:
                    target_shap = shap_vals
                
                shap_series = pd.Series(target_shap, index=proc_cols).sort_values(key=abs, ascending=False)
                top_shap = shap_series.head(8)
                
                fig_xai, ax_xai = plt.subplots(figsize=(6, 3.2))
                colors = ['#dc2626' if v > 0 else '#2563eb' for v in top_shap.values]
                y_coords = np.arange(len(top_shap))
                ax_xai.barh(y_coords, top_shap.values, color=colors, height=0.55)
                ax_xai.set_yticks(y_coords)
                ax_xai.set_yticklabels(top_shap.index, fontsize=9)
                ax_xai.set_xlabel("SHAP Impact on Predicted Stage", fontsize=10)
                ax_xai.axvline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
                ax_xai.spines['top'].set_visible(False)
                ax_xai.spines['right'].set_visible(False)
                ax_xai.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_xai)
                plt.close()
                st.caption("Red bars indicate factors increasing predicted risk; blue bars indicate protective factors.")
            except Exception as e:
                st.info(f"Local XAI visualization note: {e}")
        else:
            st.info("SHAP explainer artifact not found. Please train models in the notebook first.")

st.markdown("---")
st.caption("Notice: This clinical prediction model is developed for research and educational purposes using NHANES public health data. It does not replace diagnostic urine albumin-to-creatinine ratio (ACR) laboratory testing.")
