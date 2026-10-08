import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Heart Disease Diagnostic Dashboard", layout="wide", page_icon="🫀")

# Custom CSS for Medical Theme & 4-Tier Risk Colors
st.markdown("""
<style>
    .main { background-color: #f4f7f6; }
    .stApp { background-color: #f4f7f6; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #007bff;
        height: 100%;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #007bff; }
    .metric-label { font-size: 16px; color: #6c757d; }
    
    /* Prediction Result Boxes */
    .prediction-result {
        padding: 20px; border-radius: 10px; text-align: center;
        font-size: 24px; font-weight: bold; margin-top: 20px;
    }
    .risk-low { background-color: #28a745; color: white; }
    .risk-moderate { background-color: #ffc107; color: #212529; }
    .risk-high { background-color: #fd7e14; color: white; }
    .risk-very-high { background-color: #dc3545; color: white; }
    
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 5px;
        border: none; padding: 10px 20px; width: 100%;
    }
    .stButton>button:hover { background-color: #0056b3; color: white; }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists('models'):
        for file in os.listdir('models'):
            if file.endswith('.pkl'):
                name = file.replace('_pipeline.pkl', '').replace('_', ' ')
                models[name] = joblib.load(os.path.join('models', file))
    return models

@st.cache_resource
def load_metrics():
    if os.path.exists('models/model_metrics.json'):
        with open('models/model_metrics.json', 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_data():
    return pd.read_csv('Heart.csv', na_values=['NA', '?', ''])

models = load_models()
metrics = load_metrics()
df = load_data()

if not models:
    st.error("⚠️ No trained models found. Please run `python train_models.py` first.")
    st.stop()

# Sidebar Navigation (8 Items)
st.sidebar.title("🫀 Heart Diagnostic App")
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🩺 Patient Prediction",
        "📊 Model Performance",
        "📈 Data Visualization",
        "🧬 Feature Importance",
        "🔍 Explainability (SHAP)",
        "📁 Dataset Info",
        "⚙️ Settings"
    ]
)

# 1. Home Page
if menu == "🏠 Home":
    st.title("🏠 Heart Disease Diagnostic Dashboard")
    st.markdown("Welcome to the Heart Disease Diagnostic Dashboard. This application uses machine learning models to predict the presence of Atherosclerotic Heart Disease based on patient clinical data.")
    
    if metrics:
        best_model_name = max(metrics, key=lambda x: metrics[x]['ROC-AUC'])
        best_metrics = metrics[best_model_name]
        
        st.subheader(f"🏆 Best Model Performance: {best_model_name}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ROC-AUC", f"{best_metrics['ROC-AUC']:.2%}")
        col2.metric("Accuracy", f"{best_metrics['Accuracy']:.2%}")
        col3.metric("Recall", f"{best_metrics['Recall']:.2%}")
        col4.metric("Precision", f"{best_metrics['Precision']:.2%}")
        col5.metric("F1-Score", f"{best_metrics['F1-Score']:.2%}")

# 2. Patient Prediction
elif menu == "🩺 Patient Prediction":
    st.title("🩺 Patient Prediction")
    st.markdown("Enter the patient's clinical information below to predict the risk of Heart Disease.")
    
    model_names = list(models.keys())
    best_model = max(metrics, key=lambda x: metrics[x]['ROC-AUC'])
    selected_model = st.selectbox("Select Model", model_names, index=model_names.index(best_model))
    
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain", ["typical", "nontypical", "nonanginal", "asymptomatic"])
            rest_bp = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=120)
        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        with col3:
            ex_ang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
            ca = st.number_input("Ca (Major Vessels)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thal", ["normal", "fixed", "reversable"])
        
        submitted = st.form_submit_button("Predict Heart Disease")
        
    if submitted:
        # Map categorical inputs to numeric values expected by the model
        sex_map = {"Male": 1, "Female": 0}
        fbs_map = {"No": 0, "Yes": 1}
        rest_ecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
        ex_ang_map = {"No": 0, "Yes": 1}
        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        
        input_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex_map[sex]], 'ChestPain': [chest_pain],
            'RestBP': [rest_bp], 'Chol': [chol], 'Fbs': [fbs_map[fbs]],
            'RestECG': [rest_ecg_map[rest_ecg]], 'MaxHR': [max_hr],
            'ExAng': [ex_ang_map[ex_ang]], 'Oldpeak': [oldpeak],
            'Slope': [slope_map[slope]], 'Ca': [ca], 'Thal': [thal]
        })
        
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # ==========================================
        # 4-Tier Risk Stratification & Recommendations
        # ==========================================
        if probability < 0.25:
            risk_category = "Low Risk"
            css_class = "risk-low"
            icon = "✅"
            recommendations = """
            - **Lifestyle:** Maintain a balanced diet (e.g., Mediterranean diet) and engage in regular aerobic exercise (150 mins/week).
            - **Monitoring:** Continue with routine annual health check-ups.
            - **Prevention:** Avoid smoking and limit alcohol consumption.
            """
        elif probability < 0.50:
            risk_category = "Moderate Risk"
            css_class = "risk-moderate"
            icon = "⚠️"
            recommendations = """
            - **Medical Consultation:** Schedule a visit with your primary care physician to discuss your risk factors.
            - **Monitoring:** Regularly monitor your blood pressure, cholesterol, and blood sugar levels.
            - **Lifestyle:** Implement lifestyle modifications such as reducing sodium intake, managing stress, and increasing physical activity.
            """
        elif probability < 0.75:
            risk_category = "High Risk"
            css_class = "risk-high"
            icon = "🚨"
            recommendations = """
            - **Specialist Referral:** Schedule an appointment with a cardiologist promptly.
            - **Diagnostics:** Further diagnostic tests (e.g., ECG, stress test, echocardiogram) are highly recommended.
            - **Management:** Strictly adhere to a heart-healthy diet and follow any prescribed medications. Avoid strenuous activities until cleared by a doctor.
            """
        else:
            risk_category = "Very High Risk"
            css_class = "risk-very-high"
            icon = "🚑"
            recommendations = """
            - **Urgent Care:** **Immediate medical attention is required.** Seek urgent consultation with a cardiologist or visit the nearest emergency department.
            - **Diagnostics:** Advanced imaging and potential interventional procedures may be necessary.
            - **Warning:** Do not ignore any acute symptoms such as chest pain, shortness of breath, or dizziness.
            """
            
        # Display Risk Result
        st.markdown(f'<div class="prediction-result {css_class}">{icon} {risk_category} (Probability: {probability:.2%})</div>', unsafe_allow_html=True)
        
        # Display Recommendations
        st.markdown("### 📋 Clinical Recommendations")
        st.markdown(recommendations)
        
        # Medical Disclaimer
        st.markdown("---")
        st.caption("⚠️ *Disclaimer: This tool is for educational and demonstration purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.*")
            
        # Store in session state for SHAP analysis
        st.session_state['input_data'] = input_data
        st.session_state['selected_model'] = selected_model

# 3. Model Performance
elif menu == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")
    if metrics:
        df_metrics = pd.DataFrame(metrics).T
        st.dataframe(df_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_metrics[['ROC-AUC', 'Accuracy', 'Recall', 'Precision']].plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.title("Model Metrics Comparison")
        plt.tight_layout()
        st.pyplot(fig)

# 4. Data Visualization
elif menu == "📈 Data Visualization":
    st.title("📈 Data Visualization")
    st.markdown("Explore the distribution and relationships in the Heart Disease dataset.")
    
    tab1, tab2 = st.tabs(["Distributions", "Correlation Matrix"])
    
    with tab1:
        feature = st.selectbox("Select Feature", df.select_dtypes(include=[np.number]).columns)
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=feature, hue='AHD', kde=True, ax=ax)
        plt.title(f"Distribution of {feature} by Heart Disease")
        st.pyplot(fig)
        
    with tab2:
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)

# 5. Feature Importance
elif menu == "🧬 Feature Importance":
    st.title("🧬 Feature Importance")
    st.markdown("Global explainability: Which features drive the predictions?")
    
    tree_models = ['Random Forest', 'XGBoost', 'Decision Tree']
    available_tree_models = [m for m in tree_models if m in models]
    
    if available_tree_models:
        selected_model = st.selectbox("Select Tree-Based Model", available_tree_models)
        model = models[selected_model]
        importances = model.named_steps['classifier'].feature_importances_
            
        num_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']
        cat_features = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
        feature_names = num_features + cat_features
        
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        feat_imp.plot(kind='barh', ax=ax, color='#007bff')
        plt.title(f"Feature Importances ({selected_model})")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)

# 6. Explainability (SHAP)
# 6. Explainability (SHAP)
elif menu == "🔍 Explainability (SHAP)":
    st.title("🔍 Explainability (SHAP)")
    st.markdown("Local explainability: Understand the prediction for a specific patient.")
    
    if 'input_data' in st.session_state and 'selected_model' in st.session_state:
        selected_model = st.session_state['selected_model']
        input_data = st.session_state['input_data']
        model = models[selected_model]
        
        st.write(f"**Explaining prediction for:** {selected_model}")
        
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['classifier']
        
        X_test_trans = preprocessor.transform(input_data)
        
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_trans)
        
        # ==========================================
        # ROBUST EXTRACTION: Handle list and 3D array outputs from SHAP
        # ==========================================
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1]
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        elif len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes) -> Extract Class 1 (Disease)
            shap_values_pos = shap_values[:, :, 1]
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        else:
            shap_values_pos = shap_values
            base_val = explainer.expected_value
            
        num_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']
        cat_features = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
        feature_names = num_features + cat_features
        
        # Create the Explanation object using the correctly defined base_val
        explanation = shap.Explanation(
            values=shap_values_pos[0],
            base_values=base_val,  # <--- Now base_val is properly defined!
            data=X_test_trans[0],
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please make a prediction in the **🩺 Patient Prediction** tab first to see SHAP explanations.")

# 7. Dataset Info
elif menu == "📁 Dataset Info":
    st.title("📁 Dataset Information")
    st.markdown("""
    **Heart Disease Dataset**
    
    This dataset contains clinical parameters used to predict the presence of Atherosclerotic Heart Disease.
    
    **Features:**
    - **Age**: Patient's age in years
    - **Sex**: Biological sex (0 = Female, 1 = Male)
    - **ChestPain**: Type of chest pain (typical, nontypical, nonanginal, asymptomatic)
    - **RestBP**: Resting blood pressure (mm Hg)
    - **Chol**: Serum cholesterol (mg/dl)
    - **Fbs**: Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)
    - **RestECG**: Resting Electrocardiogram results (0, 1, 2)
    - **MaxHR**: Maximum heart rate achieved
    - **ExAng**: Exercise induced angina (0 = No, 1 = Yes)
    - **Oldpeak**: ST depression induced by exercise relative to rest
    - **Slope**: The slope of the peak exercise ST segment (1, 2, 3)
    - **Ca**: Number of major vessels colored by fluoroscopy (0-3)
    - **Thal**: Thalassemia (normal, fixed, reversable)
    
    **Target:**
    - **AHD**: Diagnosis of heart disease (0 = No, 1 = Yes)
    """)
    st.dataframe(df.head())

# 8. Settings
elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("""
    **Application Settings**
    - **Theme**: Medical (Blue/White/Clean)
    - **Models**: Pre-trained machine learning models (Random Forest, XGBoost, etc.)
    - **Framework**: Streamlit, Scikit-Learn, XGBoost, SHAP
    - **Data**: Heart.csv (Cleveland Heart Disease Dataset)
    """)