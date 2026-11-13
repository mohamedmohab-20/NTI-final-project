import os
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = ['gender', 'ethnicity', 'smoking_status']

INPUT_FEATURES_WHITELIST = [
    'age', 'gender', 'ethnicity', 'income_ratio', 'body_mass_index', 'height_cm',
    'heart_rate_bpm', 'white_blood_cells_count', 'platelets_count', 'hemoglobin_g_dl',
    'mean_corpuscular_volume_fL', 'creatinine_mg_dl', 'liver_ast_U_L', 'bilirubin_mg_dl',
    'liver_ggt_U_L', 'uric_acid_mg_dl', 'sodium_mmol_L', 'potassium_mmol_L',
    'cholesterol_mg_dl', 'alcohol_drinks_per_week', 'smoking_status'
]

STAGE_NAMES = {
    0: 'Normal',
    1: 'Microalbuminuria',
    2: 'Macroalbuminuria'
}

def load_data_from_db(db_path="databases/nhanes_1st.db", table_name="raw_dataset"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
    conn.close()
    return df

def extract_features_and_target(df):
    df_clean = df.dropna(subset=['albuminuria_risk']).copy()
    df_clean['albuminuria_risk'] = df_clean['albuminuria_risk'].astype(int)
    
    available_inputs = [c for c in INPUT_FEATURES_WHITELIST if c in df_clean.columns]
    X = df_clean[available_inputs].copy()
    y = df_clean['albuminuria_risk'].copy()
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

class ClinicalCorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.50, protected_features=None, clinical_priority=None, method='spearman'):
        self.threshold = threshold
        self.protected_features = protected_features or ['age', 'gender', 'creatinine_mg_dl', 'hemoglobin_g_dl']
        self.clinical_priority = clinical_priority or {
            'gender': 100, 'age': 100, 'creatinine_mg_dl': 100, 'hemoglobin_g_dl': 90,
            'body_mass_index': 80, 'liver_ggt_U_L': 70, 'white_blood_cells_count': 60,
            'uric_acid_mg_dl': 50, 'cholesterol_mg_dl': 50, 'income_ratio': 50,
            'heart_rate_bpm': 40, 'potassium_mmol_L': 40, 'sodium_mmol_L': 40,
            'platelets_count': 30, 'liver_ast_U_L': 20, 'mean_corpuscular_volume_fL': 10,
            'height_cm': 5
        }
        self.method = method
        self.retained_features_ = []
        self.dropped_features_ = []
        self.pruning_log_ = []
        self.corr_matrix_ = None
        self.target_corrs_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.corr_matrix_ = X_df.corr(method=self.method)
        
        self.target_corrs_ = {}
        if y is not None:
            y_s = pd.Series(y).reset_index(drop=True)
            valid_y = y_s.notnull()
            for col in X_df.columns:
                valid = valid_y & X_df[col].notnull()
                if valid.sum() > 30:
                    r, _ = stats.spearmanr(X_df.loc[valid, col], y_s.loc[valid])
                    self.target_corrs_[col] = abs(r) if not np.isnan(r) else 0.0
                else:
                    self.target_corrs_[col] = 0.0
        else:
            self.target_corrs_ = {col: 0.0 for col in X_df.columns}

        retained = list(X_df.columns)
        dropped = []
        log = []
        cols = list(X_df.columns)
        pairs = []
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                r_val = abs(self.corr_matrix_.loc[c1, c2])
                if r_val >= self.threshold:
                    pairs.append((c1, c2, r_val))

        pairs.sort(key=lambda item: item[2], reverse=True)

        for c1, c2, r_val in pairs:
            if c1 not in retained or c2 not in retained:
                continue

            c1_prot = c1 in self.protected_features
            c2_prot = c2 in self.protected_features

            if c1_prot and c2_prot:
                log.append({
                    "pair": f"{c1} vs {c2}", "corr": round(r_val, 4), "action": "KEPT_BOTH",
                    "kept": [c1, c2], "dropped": None, "reason": "Both are protected clinical anchors"
                })
                continue
            elif c1_prot and not c2_prot:
                keep_col, drop_col = c1, c2
                reason = f"{c1} is protected clinical anchor"
            elif c2_prot and not c1_prot:
                keep_col, drop_col = c2, c1
                reason = f"{c2} is protected clinical anchor"
            else:
                p1 = self.clinical_priority.get(c1, 50)
                p2 = self.clinical_priority.get(c2, 50)
                if p1 != p2:
                    keep_col, drop_col = (c1, c2) if p1 > p2 else (c2, c1)
                    reason = f"Clinical priority {keep_col} > {drop_col}"
                else:
                    t1 = self.target_corrs_.get(c1, 0.0)
                    t2 = self.target_corrs_.get(c2, 0.0)
                    keep_col, drop_col = (c1, c2) if t1 >= t2 else (c2, c1)
                    reason = f"Target association {keep_col} >= {drop_col}"

            retained.remove(drop_col)
            dropped.append(drop_col)
            log.append({
                "pair": f"{c1} vs {c2}", "corr": round(r_val, 4), "action": f"DROPPED_{drop_col}",
                "kept": keep_col, "dropped": drop_col, "reason": reason
            })

        self.retained_features_ = retained
        self.dropped_features_ = dropped
        self.pruning_log_ = log
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.retained_features_]

    def get_pruning_report(self):
        return pd.DataFrame(self.pruning_log_)

def build_preprocessor(feature_names, categorical_cols=CATEGORICAL_FEATURES, max_iter=10, random_state=42):
    cat_cols = [c for c in categorical_cols if c in feature_names]
    cont_cols = [c for c in feature_names if c not in cat_cols]

    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cont_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=max_iter, random_state=random_state)),
        ('scaler', StandardScaler())
    ])

    transformers = []
    if cont_cols:
        transformers.append(('num', cont_transformer, cont_cols))
    if cat_cols:
        transformers.append(('cat', cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False
    )
    return preprocessor, cont_cols, cat_cols

def fit_transform_pipeline(preprocessor, X_train, X_test):
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    return X_train_df, X_test_df, feature_names
