import streamlit as st
import math
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# 全局配置
st.set_page_config(
    page_title="Radiopharmaceutical Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💊"
)

# 自定义样式
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Arial', Times, serif;
        font-size: 15px;
        color: #212529;
        background-color: #f8f9fa;
    }
    h1 {
        color: #1a476f;
        font-weight: bold;
        font-size: 26px;
        border-bottom: 3px solid #2c5282;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2 {
        color: #2c5282;
        font-weight: 600;
        font-size: 22px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .model-card {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #dee2e6;
        margin-bottom: 15px;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .model-card.selected {
        border: 2px solid #2c5282;
        background-color: #e8f4f8;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px 12px;
    }
    .stButton > button {
        background-color: #2c5282;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1a365d;
        transform: translateY(-1px);
    }
    .result-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #2c5282;
        margin: 15px 0;
    }
    .eval-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #28a745;
        margin: 15px 0;
    }
    .eval-card.warning {
        border-left: 4px solid #dc3545;
    }
    .dataframe {
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }
    .divider {
        border-top: 1px solid #dee2e6;
        margin: 20px 0;
    }
    .history-item {
        padding: 10px;
        border-bottom: 1px solid #f1f3f5;
    }
    .history-item:last-child {
        border-bottom: none;
    }
    [data-testid="stSidebar"] {
        background-color: #f1f3f5;
        padding-top: 2rem;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# 分布预测模块
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Core Feature Model"
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# 剂量计算模块
if "calculation_history" not in st.session_state:
    st.session_state.calculation_history = []
if "hed_calculation_history" not in st.session_state:
    st.session_state.hed_calculation_history = []

# 侧边栏导航
st.sidebar.title("📋 Function Navigation")
selected_module = st.sidebar.selectbox(
    "Select Function Module",
    options=[
        "Radiopharmaceutical Biodistribution Prediction",
        "Radioactive Drug Injection Dose Calculator"
    ],
    index=0
)

# 公共函数（分布预测模块）
@st.cache_resource
def load_model(prefix):
    try:
        model = joblib.load(f'{prefix}_model.pkl')
        num_imputer = joblib.load(f'{prefix}_num_imputer.pkl')
        scaler = joblib.load(f'{prefix}_scaler.pkl')
        te = joblib.load(f'{prefix}_te.pkl')
        selected_feats = joblib.load(f'{prefix}_selected_feats.pkl')
        num_cols = joblib.load(f'{prefix}_num_cols.pkl')
        feature_names = joblib.load(f'{prefix}_feature_names.pkl')
        cat_unique_vals = joblib.load(f'{prefix}_cat_unique_vals.pkl')
        return model, num_imputer, scaler, te, selected_feats, num_cols, feature_names, cat_unique_vals
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e}. Please run the training scripts first.")
        return None

def select_model(model_name):
    st.session_state.selected_model = model_name
    st.rerun()

def optimize_dosage(input_dict, cat_cols, num_cols_input, num_imputer, scaler, te, selected_feats, model, time_val,
                    threshold):
    dosage_col = "Injection Dosage"
    if dosage_col not in num_cols_input:
        st.warning("Injection dose field not found, dose optimization cannot be performed.")
        return None, []
    dosage_attempts = []
    recommended_dosage = None
    for dosage in np.arange(0.5, 10.1, 0.5):
        new_input = input_dict.copy()
        new_input[dosage_col] = dosage
        new_input_df = pd.DataFrame([new_input])
        X_cat_new = new_input_df[cat_cols].copy()
        X_num_new = new_input_df[num_cols_input].copy()
        for col in X_num_new.columns:
            X_num_new[col] = pd.to_numeric(X_num_new[col], errors='coerce')
        X_num_imputed_new = pd.DataFrame(num_imputer.transform(X_num_new), columns=X_num_new.columns)
        X_num_scaled_new = pd.DataFrame(scaler.transform(X_num_imputed_new), columns=X_num_new.columns)
        current_num_cols = num_cols_input
        for i in range(len(current_num_cols)):
            for j in range(i + 1, min(i + 3, len(current_num_cols))):
                col1, col2 = current_num_cols[i], current_num_cols[j]
                if col1 in X_num_scaled_new.columns and col2 in X_num_scaled_new.columns:
                    X_num_scaled_new[f'{col1}_mul_{col2}'] = X_num_scaled_new[col1] * X_num_scaled_new[col2]
        for col in current_num_cols[:3]:
            if col in X_num_scaled_new.columns:
                X_num_scaled_new[f'{col}_sq'] = X_num_scaled_new[col] ** 2
                X_num_scaled_new[f'{col}_log'] = np.log1p(np.abs(X_num_scaled_new[col]))
        if len(current_num_cols) >= 2:
            c1, c2 = current_num_cols[0], current_num_cols[1]
            if c1 in X_num_scaled_new.columns and c2 in X_num_scaled_new.columns:
                X_num_scaled_new[f'{c1}_div_{c2}'] = X_num_scaled_new[c1] / (X_num_scaled_new[c2] + 1e-6)
        X_cat_new = X_cat_new.fillna('NA').astype(str)
        X_cat_encoded_new = te.transform(X_cat_new)
        X_processed_new = pd.concat([X_cat_encoded_new, X_num_scaled_new], axis=1)
        X_processed_new = X_processed_new.fillna(0)
        for feat in selected_feats:
            if feat not in X_processed_new.columns:
                X_processed_new[feat] = 0
        X_final_new = X_processed_new[selected_feats]
        new_pred = model.predict(X_final_new)[0]
        meets_criteria = new_pred > threshold
        dosage_attempts.append({'Dosage': round(dosage, 1), 'Prediction Value': round(new_pred, 4), 'Meets Standard': meets_criteria})
        if meets_criteria and recommended_dosage is None:
            recommended_dosage = dosage
    return recommended_dosage, dosage_attempts

#模块1：分布预测
if selected_module == "Radiopharmaceutical Biodistribution Prediction":
    st.title("Radiopharmaceutical Biodistribution Prediction Platform")
    st.markdown("*A machine learning web page based on radiopharmaceutical properties to predict tumor biodistribution in mice*")

    st.subheader("1. Model Selection")
    col_model1, col_model2 = st.columns(2, gap="large")
    with col_model1:
        is_selected1 = st.session_state.selected_model == "Core Feature Model"
        card_class = "model-card selected" if is_selected1 else "model-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3>Core Feature Model</h3>
            <p style='color:#6c757d;'>Basic feature set with XGBoost regression</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select This Model", key="btn1"):
            select_model("Core Feature Model")
    with col_model2:
        is_selected2 = st.session_state.selected_model == "Extended Feature Model"
        card_class = "model-card selected" if is_selected2 else "model-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3>Extended Feature Model</h3>
            <p style='color:#6c757d;'>Chelator-enhanced features with XGBoost regression</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select This Model", key="btn2"):
            select_model("Extended Feature Model")

    st.markdown(f"""
    <div style='color:#2c5282; font-weight:600; margin:10px 0;'>
        Current Selected Model: {st.session_state.selected_model}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.selected_model == "Core Feature Model":
        prefix = "xgboost"
    elif st.session_state.selected_model == "Extended Feature Model":
        prefix = "efm_xgboost"
    else:
        prefix = "xgboost"
    components = load_model(prefix)

    if components:
        model, num_imputer, scaler, te, selected_feats, num_cols, feature_names, cat_unique_vals = components
        if st.session_state.selected_model == "Core Feature Model":
            n_cat = 6
            n_num = 6
        else:
            n_cat = 7
            n_num = 6
        cat_cols = feature_names[:n_cat]
        num_cols_input = feature_names[n_cat: n_cat + n_num]

        st.subheader("2. Input Parameters")
        input_dict = {}
        col_cat, col_num = st.columns(2, gap="large")
        with col_cat:
            st.markdown("<h3 style='font-size:16px;'>Categorical Features</h3>", unsafe_allow_html=True)
            for i, col in enumerate(cat_cols):
                options = cat_unique_vals.get(col, [])
                input_dict[col] = st.selectbox(label=f"{col}", options=options, key=f"cat_{i}")
        with col_num:
            st.markdown("<h3 style='font-size:16px;'>Numerical Features</h3>", unsafe_allow_html=True)
            for i, col in enumerate(num_cols_input):
                val_str = st.text_input(label=f"{col}", value="NA", key=f"num_{i}")
                if val_str.strip().upper() == "NA":
                    input_dict[col] = np.nan
                else:
                    try:
                        input_dict[col] = float(val_str)
                    except ValueError:
                        st.warning(f"Invalid input for {col} - using NA instead")
                        input_dict[col] = np.nan

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("3. Prediction & Results")
        if st.button("Run Prediction", type="primary"):
            with st.spinner('Processing data...'):
                input_df = pd.DataFrame([input_dict])
                st.markdown("<h3 style='font-size:16px;'>User Input Data</h3>", unsafe_allow_html=True)
                input_display = input_df.copy().replace({np.nan: "NA"})
                st.dataframe(input_display.style.set_properties(**{'background-color': 'white', 'border': '1px solid #dee2e6'}), use_container_width=True)

                X_cat = input_df[cat_cols].copy()
                X_num = input_df[num_cols_input].copy()
                for col in X_num.columns:
                    X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
                X_num_imputed = pd.DataFrame(num_imputer.transform(X_num), columns=X_num.columns)
                X_num_scaled = pd.DataFrame(scaler.transform(X_num_imputed), columns=X_num.columns)

                current_num_cols = num_cols
                for i in range(len(current_num_cols)):
                    for j in range(i + 1, min(i + 3, len(current_num_cols))):
                        col1, col2 = current_num_cols[i], current_num_cols[j]
                        if col1 in X_num_scaled.columns and col2 in X_num_scaled.columns:
                            X_num_scaled[f'{col1}_mul_{col2}'] = X_num_scaled[col1] * X_num_scaled[col2]
                for col in current_num_cols[:3]:
                    if col in X_num_scaled.columns:
                        X_num_scaled[f'{col}_sq'] = X_num_scaled[col] ** 2
                        X_num_scaled[f'{col}_log'] = np.log1p(np.abs(X_num_scaled[col]))
                if len(current_num_cols) >= 2:
                    c1, c2 = current_num_cols[0], current_num_cols[1]
                    if c1 in X_num_scaled.columns and c2 in X_num_scaled.columns:
                        X_num_scaled[f'{c1}_div_{c2}'] = X_num_scaled[c1] / (X_num_scaled[c2] + 1e-6)

                X_cat = X_cat.fillna('NA').astype(str)
                X_cat_encoded = te.transform(X_cat)
                X_processed = pd.concat([X_cat_encoded, X_num_scaled], axis=1).fillna(0)
                for feat in selected_feats:
                    if feat not in X_processed.columns:
                        X_processed[feat] = 0
                X_final = X_processed[selected_feats]
                prediction = model.predict(X_final)[0]

                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model': st.session_state.selected_model,
                    'prediction': round(prediction, 4),
                    'input_data': input_dict.copy()
                }
                st.session_state.prediction_history.insert(0, history_entry)
                if len(st.session_state.prediction_history) > 10:
                    st.session_state.prediction_history = st.session_state.prediction_history[:10]

                st.markdown("""
                <div class="result-card">
                    <h3 style='margin:0;'>Prediction Result</h3>
                    <p style='font-size:18px; font-weight:bold; color:#2c5282; margin:10px 0;'>
                        {:.4f}
                    </p>
                    <p style='color:#6c757d; margin:0;'>
                        Model: {} | Calculation Time: {}
                    </p>
                </div>
                """.format(prediction, st.session_state.selected_model, datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

                model_trigger_list = ["Core Feature Model", "Extended Feature Model"]
                if st.session_state.selected_model in model_trigger_list:
                    nuclide_val = target_val = time_val = chelating_agent_val = ""
                    for k, v in input_dict.items():
                        k_lower = str(k).lower()
                        if k_lower == "nuclide":
                            nuclide_val = str(v).strip()
                        if "target" in k_lower:
                            target_val = str(v).strip()
                        if "time" in k_lower and pd.notna(v):
                            time_val = str(int(float(v))).strip()
                        if "chelating agent" in k_lower or k_lower == "chelatingagent" or k_lower == "chelating":
                            chelating_agent_val = str(v).strip()

                    condition_met = False
                    if st.session_state.selected_model == "Core Feature Model":
                        condition_met = (nuclide_val == "177Lu" and target_val == "PSMA" and time_val in ["1", "4", "24"])
                    elif st.session_state.selected_model == "Extended Feature Model":
                        condition_met = (nuclide_val == "177Lu" and target_val == "PSMA" and chelating_agent_val == "DOTA" and time_val in ["1", "4", "24"])

                    if condition_met:
                        threshold_map = {"1": 14.5, "4": 18.1, "24": 13.8}
                        threshold = threshold_map[time_val]
                        rec_dosage, dosage_attempts = optimize_dosage(input_dict, cat_cols, num_cols_input, num_imputer, scaler, te, selected_feats, model, time_val, threshold)
                        original_dosage = input_dict.get("Injection Dosage", "NA")

                        if prediction > threshold:
                            eval_result = "Excellent"
                            eval_remark = f"Current dose({original_dosage}),{time_val}h Uptake（{prediction:.4f}）,better than Lu-177-PSMA617（threshold：{threshold}）。"
                            card_class = "eval-card"
                        else:
                            eval_result = "Not up to standard"
                            eval_remark = f"Current dose({original_dosage}),{time_val}h Uptake（{prediction:.4f}）,not up to standard（threshold：{threshold}）。"
                            card_class = "eval-card warning"

                        st.markdown(f"""
                        <div class="{card_class}">
                            <h3 style='margin:0;'>PSMA617 Excellence Standard Evaluation</h3>
                            <p style='font-size:16px; font-weight:bold; margin:10px 0;'>
                                Evaluation: {eval_result}
                            </p>
                            <p style='font-size:14px; color:#4a5568; margin:10px 0;'>
                                Remark: {eval_remark}
                            </p>
                            <p style='color:#6c757d; margin:0;'>
                                Evaluation Standard: {time_val}h tumor uptake ≥ {threshold} | Model: {st.session_state.selected_model}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if dosage_attempts:
                            st.markdown("<h3 style='font-size:16px;'>Dosage Optimization Attempts (Full Range)</h3>", unsafe_allow_html=True)
                            dosage_df = pd.DataFrame(dosage_attempts)
                            dosage_df['Meets Standard'] = dosage_df['Meets Standard'].map({True: '✅', False: '❌'})
                            st.dataframe(dosage_df.style.set_properties(**{'background-color': 'white', 'border': '1px solid #dee2e6'}), use_container_width=True, hide_index=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("4. Recent Prediction History (Last 10)")
        if st.session_state.prediction_history:
            history_data = []
            for entry in st.session_state.prediction_history:
                input_str = ", ".join([f"{k}: {v if not pd.isna(v) else 'NA'}" for k, v in entry['input_data'].items()])
                history_data.append({
                    'Time': entry['timestamp'],
                    'Model': entry['model'],
                    'Prediction Value': entry['prediction'],
                    'Input Summary': input_str[:100] + "..." if len(input_str) > 100 else input_str
                })
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df.style.set_properties(**{'background-color': 'white', 'border': '1px solid #dee2e6'}), use_container_width=True, hide_index=True)
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.markdown("<p style='color:#6c757d;'>No prediction history yet</p>", unsafe_allow_html=True)
    else:
        st.error("Failed to load model components. Please check model files.")
        st.stop()

    st.markdown("""
    <div style='margin-top:50px; padding-top:20px; border-top:1px solid #dee2e6; color:#6c757d; text-align:center;'>
        Radiopharmaceutical Biodistribution Prediction Platform | Designed for Academic Standards
    </div>
    """, unsafe_allow_html=True)

# 模块2：放射性药物计算器
elif selected_module == "Radioactive Drug Injection Dose Calculator":
    FIXED_DOSE_DRUGS = {
        "Custom Fixed": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1.0},
        "¹⁷⁷Lu-PSMA-617": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1000.0},
        "¹⁷⁷Lu‑DOTATATE": {"physical_half_life": 6.647, "fixed_total_activity": 7400.0, "calibration_concentration": 1000.0},
        "¹³¹I‑MIBG": {"physical_half_life": 8.02, "fixed_total_activity": 11100.0, "calibration_concentration": 111.0},
        "⁹⁹ᵐTc‑MDP": {"physical_half_life": 0.2503, "fixed_total_activity": 740.0, "calibration_concentration": 37.0},
        "⁹⁹ᵐTc‑DTPA": {"physical_half_life": 0.2503, "fixed_total_activity": 555.0, "calibration_concentration": 37.0},
        "⁶⁸Ga‑DOTATATE": {"physical_half_life": 0.047, "fixed_total_activity": 150.0, "calibration_concentration": 218.0},
    }
    WEIGHT_BASED_DRUGS = {
        "Custom Parameters": {"physical_half_life": 11.4, "target_dose_per_kg": 0.055, "calibration_concentration": 1.1},
        "²²³RaCl₂": {"physical_half_life": 11.4, "target_dose_per_kg": 0.055, "calibration_concentration": 1.1},
        "²²⁵Ac-PSMA-617": {"physical_half_life": 9.92, "target_dose_per_kg": 0.125, "calibration_concentration": 1.1},
        "²²⁵Ac-DOTATATE": {"physical_half_life": 9.92, "target_dose_per_kg": 0.120, "calibration_concentration": 1.1},
        "²²⁵Ac-PSMA-I&T": {"physical_half_life": 9.92, "target_dose_per_kg": 0.100, "calibration_concentration": 1.1},
        "²²⁵Ac-J591": {"physical_half_life": 9.92, "target_dose_per_kg": 0.035, "calibration_concentration": 1.1},
        "²¹²Pb-DOTATATE": {"physical_half_life": 0.4433, "target_dose_per_kg": 2.5012, "calibration_concentration": 1.1}
    }
    ORGAN_DOSE_TABLE2 = {
        "Lacrimal Gland": 2.1, "Salivary Gland": 0.63, "Kidney": 0.43, "Left Colon": 0.58, "Rectum": 0.56,
        "Right Colon": 0.32, "Bladder Wall": 0.32, "Heart Wall": 0.17, "Liver": 0.09, "Lung": 0.11, "Whole Body": 0.037
    }
    MOUSE_TUMOR_DOSE = {
        "Lacrimal Gland": 24.0, "Salivary Gland": 8.5, "Kidney": 5.8, "Left Colon": 7.2, "Rectum": 6.9,
        "Right Colon": 4.1, "Bladder Wall": 4.1, "Heart Wall": 4.3, "Liver": 2.2, "Lung": 1.8, "Whole Body": 1.4, "Tumor": 30.0
    }

    st.title("💉 Radioactive Drug Calculator")
    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)
    st.subheader("🧪 Extrapolation of Tumor‑Bearing Mouse Dose to Human Equivalent Dose (HED)")
    st.divider()

    hed_method = st.selectbox(
        "Select Extrapolation Method",
        options=[
            "Method 1: Custom BSA Universal Formula",
            "Method 2: Allometric Scaling Law (b=0.67)",
            "Method 3: Tumor Volume Normalization"
        ],
        index=0
    )
    st.caption("📌 Unit: Mouse Dose(MBq/kg) | HED(MBq/kg)")
    hed_result = 0

    with st.expander("📝 Input Calculation Parameters", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            mouse_dose = st.number_input("Mouse Dose", min_value=0.001, value=12.0, step=0.1, key="mouse_dose")
            mouse_weight = st.number_input("Mouse Weight (g)", min_value=10.0, value=20.0, step=1.0, key="mouse_weight")
        with col_b:
            human_weight = st.number_input("Human Weight (kg)", min_value=1.0, value=60.0, step=1.0, key="human_weight")
            human_height = st.number_input("Human Height (cm)", min_value=100.0, value=170.0, step=1.0, key="human_height")
        if hed_method == "Method 3: Tumor Volume Normalization":
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                mouse_tumor = st.number_input("Mouse Tumor Volume (mm³)", min_value=10.0, value=200.0, step=10.0)
            with col_t2:
                human_tumor = st.number_input("Human Tumor Volume (mm³)", min_value=100.0, value=3000.0, step=100.0)

    calc_hed = st.button("🧮 Calculate HED", type="primary", use_container_width=True)
    if calc_hed:
        mouse_w_kg = mouse_weight / 1000
        if hed_method == "Method 1: Custom BSA Universal Formula":
            human_bsa = 0.007184 * (human_weight ** 0.425) * (human_height ** 0.725)
            mouse_bsa = 0.1 * (mouse_w_kg ** 0.67)
            hed_result = mouse_dose * (mouse_bsa / human_bsa)
        elif hed_method == "Method 2: Allometric Scaling Law (b=0.67)":
            hed_result = mouse_dose * ((mouse_w_kg / human_weight) ** 0.67)
        elif hed_method == "Method 3: Tumor Volume Normalization":
            hed_result = mouse_dose * (mouse_tumor / human_tumor)
        total_injection_mbq = hed_result * human_weight
        hed_record = {
            "Calculation Time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "Method": hed_method.replace("Method 1: ", "").replace("Method 2: ", "").replace("Method 3: ", ""),
            "Mouse Dose(MBq/kg)": round(mouse_dose, 2),
            "Mouse Weight(g)": round(mouse_weight, 1),
            "Human Weight(kg)": round(human_weight, 1),
            "Human Height(cm)": round(human_height, 1),
            "HED(MBq/kg)": round(hed_result, 4),
            "Total Dose(MBq)": round(total_injection_mbq, 4)
        }
        st.session_state.hed_calculation_history.insert(0, hed_record)
        if len(st.session_state.hed_calculation_history) > 10:
            st.session_state.hed_calculation_history.pop()

    st.subheader("📊 HED Results")
    res_hed1, res_hed2, res_hed3 = st.columns(3)
    if calc_hed:
        total_injection_mbq = hed_result * human_weight
        res_hed1.metric("Method", hed_method.replace("Method 1: ", "").replace("Method 2: ", "").replace("Method 3: ", ""))
        res_hed2.metric("Human Equivalent Dose (HED)", f"{hed_result:.4f} MBq/kg")
        res_hed3.metric("Total Recommended Dose", f"{total_injection_mbq:.4f} MBq")
    else:
        st.info("👆 Enter parameters and click [Calculate HED] to get results")

    st.subheader("📜 Recent 10 HED Calculation Records")
    if st.session_state.hed_calculation_history:
        st.dataframe(st.session_state.hed_calculation_history, use_container_width=True, hide_index=True)
    else:
        st.info("No HED calculation records yet")
    st.divider()

    st.markdown("<div style='margin:100px 0'></div>", unsafe_allow_html=True)
    st.subheader("⚙️  Injection Dose Calculation")
    st.divider()
    dose_type = st.selectbox("Select Dose Calculation Type", options=["Fixed Dose", "Weight‑Based Dose"], index=0)
    if dose_type == "Fixed Dose":
        selected_drug = st.selectbox("Select Fixed Dose Drug", options=list(FIXED_DOSE_DRUGS.keys()), index=1)
        preset = FIXED_DOSE_DRUGS[selected_drug]
    else:
        selected_drug = st.selectbox("Select Weight‑Based Drug", options=list(WEIGHT_BASED_DRUGS.keys()), index=0)
        preset = WEIGHT_BASED_DRUGS[selected_drug]

    st.caption("📌 Half‑life Formula: 1/Effective = 1/Physical + 1/Biological")
    col_phys, col_bio, col_eff = st.columns(3)
    with col_phys:
        physical_half_life = st.number_input("Physical Half‑life (days)", min_value=0.1, value=preset["physical_half_life"], step=0.1)
    with col_bio:
        biological_half_life = st.number_input("Biological Half‑life (days)", min_value=0.1, value=20.0, step=0.1)
    with col_eff:
        effective_half_life = 1 / (1 / physical_half_life + 1 / biological_half_life) if physical_half_life > 0 and biological_half_life > 0 else 0.0
        st.number_input("Effective Half‑life (days)", value=round(effective_half_life, 4), disabled=False)

    col3, col4 = st.columns(2)
    with col3:
        weight = st.number_input("Patient Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=1.0)
    with col4:
        time_elapsed = st.number_input("Time Elapsed Since Calibration (days)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)

    col5, col6 = st.columns(2)
    with col5:
        calibration_concentration = st.number_input("Calibration Concentration (MBq/mL)", min_value=0.001, max_value=100000.0, value=preset["calibration_concentration"], step=0.0001, format="%.4f")
    with col6:
        if dose_type == "Fixed Dose":
            total_recommended_activity_fixed = st.number_input("Fixed Total Activity (MBq)", min_value=10.0, value=preset["fixed_total_activity"], step=0.0001, format="%.4f")
        else:
            target_dose_per_kg = st.number_input("Target Dose (MBq/kg)", min_value=0.001, max_value=5.0, value=preset["target_dose_per_kg"], step=0.0001, format="%.4f")

    calc_button = st.button("🧮 Calculate Injection Dose", type="primary", use_container_width=True)
    total_recommended_activity = decay_coefficient = injection_volume = total_activity_gbq = 0.0
    if calc_button:
        if dose_type == "Fixed Dose":
            total_recommended_activity = total_recommended_activity_fixed
        else:
            total_recommended_activity = weight * target_dose_per_kg
        lambda_decay = math.log(2) / physical_half_life
        decay_coefficient = math.exp(-lambda_decay * time_elapsed)
        injection_volume = total_recommended_activity / (decay_coefficient * calibration_concentration)
        total_activity_gbq = total_recommended_activity / 1000
        record = {
            "Calculation Time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "Drug Name": selected_drug,
            "Dose Type": dose_type,
            "Physical Half‑life(d)": round(physical_half_life, 2),
            "Total Activity(MBq)": round(total_recommended_activity, 4),
            "Total Activity(GBq)": round(total_activity_gbq, 4),
            "Decay Coefficient": round(decay_coefficient, 4),
            "Injection Volume(mL)": round(injection_volume, 4)
        }
        st.session_state.calculation_history.insert(0, record)
        if len(st.session_state.calculation_history) > 10:
            st.session_state.calculation_history.pop()

    st.subheader("📊 Injection Dose Results")
    if calc_button:
        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
        res_col1.metric("Drug", selected_drug)
        res_col2.metric("Dose Type", dose_type)
        res_col3.metric("Total Activity(MBq)", f"{total_recommended_activity:.4f}")
        res_col4.metric("Decay Coefficient", f"{decay_coefficient:.4f}")
        res_col5.metric("Injection Volume(mL)", f"{injection_volume:.4f} mL")
    else:
        st.info("👆 Set parameters and click [Calculate Injection Dose] to get results")

    if calc_button and selected_drug == "¹⁷⁷Lu-PSMA-617" and dose_type == "Fixed Dose":
        st.divider()
        st.subheader("🧬 ¹⁷⁷Lu‑PSMA‑617 Organ Absorbed Dose Comparison")
        st.caption(f"📌 Based on Total Activity: {total_activity_gbq:.3f} GBq | Unit: Absorbed Dose(Gy)")
        compare_data = []
        for organ in ORGAN_DOSE_TABLE2.keys():
            human_dose = ORGAN_DOSE_TABLE2[organ] * total_activity_gbq
            mouse_dose = MOUSE_TUMOR_DOSE[organ] * total_activity_gbq
            compare_data.append({
                "Organ": organ,
                "Human(Gy/GBq)": ORGAN_DOSE_TABLE2[organ],
                "Mouse(Gy/GBq)": MOUSE_TUMOR_DOSE[organ],
                "Human Dose(Gy)": round(human_dose, 4),
                "Mouse Dose(Gy)": round(mouse_dose, 4)
            })
        compare_data.append({
            "Organ": "Tumor",
            "Human(Gy/GBq)": "-",
            "Mouse(Gy/GBq)": MOUSE_TUMOR_DOSE["Tumor"],
            "Human Dose(Gy)": "-",
            "Mouse Dose(Gy)": round(MOUSE_TUMOR_DOSE["Tumor"] * total_activity_gbq, 4)
        })
        df = pd.DataFrame(compare_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("<div style='margin:60px 0'></div>", unsafe_allow_html=True)
    st.subheader("⏱️ In Vivo Residual Activity Calculation")
    st.divider()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        post_injection_time = st.number_input("Time After Injection", min_value=0.0, value=24.0, step=1.0)
        time_unit = st.selectbox("Time Unit", ["Hours (h)", "Days (d)"], index=0)
    t_h = post_injection_time * 24 if time_unit == "Days (d)" else post_injection_time
    if calc_button and total_recommended_activity > 0:
        lambda_eff = math.log(2) / effective_half_life
        remaining_activity_mbq = total_recommended_activity * math.exp(-lambda_eff * (t_h / 24))
        remaining_percent = remaining_activity_mbq / total_recommended_activity * 100
        st.subheader("📈 Residual Activity Results")
        rem1, rem2, rem3, rem4 = st.columns(4)
        rem1.metric("Time After Injection", f"{post_injection_time} {time_unit}")
        rem2.metric("Initial Total Activity", f"{total_recommended_activity:.4f} MBq")
        rem3.metric("Residual Activity", f"{remaining_activity_mbq:.4f} MBq")
        rem4.metric("Residual Ratio", f"{remaining_percent:.4f} %")
    else:
        st.info("👆 Complete Step 1 first to calculate residual activity")

    st.divider()
    st.subheader("📜 Recent 10 Injection Dose Records")
    if st.session_state.calculation_history:
        st.dataframe(st.session_state.calculation_history, use_container_width=True, hide_index=True)
    else:
        st.info("No injection calculation records yet")

    # st.divider()
    # st.subheader("📐 Calculation Formulas")
    # st.markdown("##### 🧪 HED Extrapolation Formulas")
    # st.latex(r"HED = D_{mouse} \times \dfrac{BSA_{mouse}}{BSA_{human}}")
    # st.latex(r"HED = D_{mouse} \times \left( \dfrac{W_{mouse}}{W_{human}} \right)^{0.67}")
    # st.latex(r"HED = D_{mouse} \times \dfrac{TV_{mouse}}{TV_{human}}")
    # st.divider()
    # st.markdown("##### ⚛️ Half‑life & Injection Dose Formulas")
    # st.markdown("**Half‑life Relationship**")
    # st.latex(r"\frac{1}{t_{eff}} = \frac{1}{t_{phys}} + \frac{1}{t_{bio}}")
    # st.markdown("**Decay Coefficient & Injection Volume**")
    # st.latex(r"k = e^{-\frac{\ln2}{t_{phys}} \times t}")
    # st.latex(r"\text{Injection Volume(mL)} = \frac{\text{Total Activity(MBq)}}{k \times \text{Conc.(MBq/mL)}}")
    # st.markdown("**Residual Activity in Vivo**")
    # st.latex(r"A_t(\text{MBq}) = A_0(\text{MBq}) \cdot e^{-\frac{\ln2}{t_{eff}} \cdot t}")

    st.divider()
    st.warning("""
    ⚠️ Radiation Safety & Academic‑Use Disclaimer
    1. This calculator serves academic research, pre‑clinical studies and clinical dosimetry assistance only. It shall not be the sole basis for radiopharmaceutical administration; final clinical decisions must be determined by qualified nuclear medicine physicians.
    2. Physical half‑life is an intrinsic radionuclide constant, while biological half‑life is an empirical reference affected by individual metabolism, pathology and organ function.
    3. Preset radiopharmaceutical doses are literature‑based references. Clinical application must comply with official drug specifications, clinical guidelines and institutional standards.
    4. Total activity, injection volume, decay correction and human equivalent dose (HED) shall be verified before clinical administration following radiation protection protocols.
    5. HED extrapolation, organ‑specific dose and residual activity are theoretical approximations, without accounting for tumour heterogeneity, radiosensitivity and inter‑species physiological differences.
    6. Radiopharmaceuticals emit ionising radiation. All relevant operations shall follow national radiological regulations to protect patients, medical staff and the public.
    7. Decay correction is calculated via pure physical kinetics, excluding biological clearance, non‑specific binding and excretion, thus involving inherent computational uncertainties.
    """)
