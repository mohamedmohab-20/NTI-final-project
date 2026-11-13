# Intelligent Kidney Health Assessment System
### NHANES 3-Class Ordinal Classification Pipeline

This repository contains an end-to-end machine learning and deep learning system for predicting kidney risk stages directly from routine, non-invasive clinical biomarkers without data leakage, following KDIGO (Kidney Disease: Improving Global Outcomes) clinical guidelines.

## Clinical Staging (Target)
The target label is derived from the urinary Albumin-to-Creatinine Ratio (ACR):
- **Class 0 (Normal)**: ACR < 30 mg/g (KDIGO Stage A1)
- **Class 1 (Microalbuminuria)**: 30 <= ACR < 300 mg/g (KDIGO Stage A2)
- **Class 2 (Macroalbuminuria)**: ACR >= 300 mg/g (KDIGO Stage A3)

Urinary ACR and spot urine albumin/creatinine measurements are excluded from the model predictors to prevent data leakage.

## Project Structure
```
.
├── databases/
│   ├── nhanes_1st.db                  # Raw and harmonized multi-cycle NHANES SQLite database
│   └── SQL_queries/                  # SQL queries for multi-table joins and target definitions
├── etl/                              # Data collection and harmonization pipeline
│   ├── 01_ingestion.py               # Raw SAS XPT extraction into SQLite
│   ├── 02_harmonization.py           # Clinical standardization and biological bounds
│   ├── 03_transformation.py          # Tensor preparation
│   ├── etl_pipeline.py               # Callable ETL summary and verification module
│   └── ELT_Config.json               # Configurations, biological bounds, and clinical thresholds
├── preprocessing.py                  # Database loading, correlation filtering, MICE imputation, and scaling
├── model.py                          # PyTorch Ordinal MLP architecture, training routines, and ONNX exporter
├── kidney_health_assessment.ipynb    # Executable master analysis notebook
├── app.py                            # Minimalist Streamlit web interface with ONNX and XAI
├── kidney_model.onnx                 # Exported champion model in ONNX format
├── preprocessor.pkl                  # Serialized MICE imputation and scaling pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Methodology Summary
1. **Data Collection & Lineage**: Multi-cycle NHANES survey data ingested and harmonized into `nhanes_1st.db`.
2. **Clinical Preprocessing & Redundancy Filtering**:
   - Extraction of 21 routine biomarkers.
   - Clinical correlation filtering using training-set Spearman correlation (`ClinicalCorrelationFilter`), protecting essential anchors (`age`, `gender`, `creatinine_mg_dl`, `hemoglobin_g_dl`) while pruning collinear features (`height_cm`, etc.).
   - MICE statistical imputation (`IterativeImputer`) and standardization (`StandardScaler`) fitted strictly on training data.
   - One-hot encoding for categorical variables (`gender`, `ethnicity`, `smoking_status`).
3. **Machine Learning Baselines**: 5 diverse models trained with class weighting:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
4. **Fine-Tuning & Comparison**: Top 2 ML models tuned using 3-fold Stratified Cross-Validation targeting Macro F1, evaluated against baselines with comparative performance charts.
5. **Deep Learning Approach**: PyTorch `OrdinalMLP` trained with cumulative ordinal binary heads and positive class weighting, enforcing monotonicity ($P(\text{Stage} \ge 2) \le P(\text{Stage} \ge 1)$).
6. **Explainable AI (XAI)**: SHAP analysis for global feature importance and individual patient risk factor attribution.
7. **Production Deployment**: Exported to ONNX (`kidney_model.onnx`) and deployed via a minimalist Streamlit dashboard (`app.py`).

## How to Launch and Try the Web Interface

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
Execute the following command in your terminal:
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### 3. Using the Web Interface
1. Enter patient clinical values in the input form (Age, Gender, Body Mass Index, Serum Creatinine, Hemoglobin, Uric Acid, etc.).
2. Click **Run Clinical Risk Assessment**.
3. View the predicted KDIGO risk classification, class probability distribution bars, and local biomarker SHAP attributions explaining the primary risk drivers.
