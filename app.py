# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="🧒 Kids Screen Time Health Impact Predictor",
    page_icon="🖥️",
    layout="wide"
)

# --- Load Model and Preprocessor Artifacts ---
@st.cache_resource
def load_artifacts(model_type='ensemble'):
    """Loads all necessary artifacts for prediction."""
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None, None, None, None
    try:
        model_pipeline = joblib.load(os.path.join(model_dir, f'{model_type}_model.joblib'))
        mlb = joblib.load(os.path.join(model_dir, 'multilabel_binarizer.joblib'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
        
        explainers = {}
        if model_type == 'ensemble':
            xgb_component = joblib.load(os.path.join(model_dir, 'ensemble_xgb_component.joblib'))
            lgbm_component = joblib.load(os.path.join(model_dir, 'ensemble_lgbm_component.joblib'))
            cat_component = joblib.load(os.path.join(model_dir, 'ensemble_cat_component.joblib'))
            
            xgb_models = xgb_component.named_steps['model'].estimators_
            lgbm_models = lgbm_component.named_steps['model'].estimators_
            cat_models = cat_component.named_steps['model'].estimators_

            explainers['XGBoost Component'] = [shap.TreeExplainer(model) for model in xgb_models]
            explainers['LightGBM Component'] = [shap.TreeExplainer(model) for model in lgbm_models]
            explainers['CatBoost Component'] = [shap.TreeExplainer(model) for model in cat_models]

        else:
            fitted_models = model_pipeline.named_steps['model'].estimators_
            explainers[model_type.upper()] = [shap.TreeExplainer(model) for model in fitted_models]
        
        return model_pipeline, mlb, feature_names, explainers
        
    except FileNotFoundError:
        st.error(f"Artifacts for '{model_type}' not found. Please train it first by running main.py.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        return None, None, None, None

# --- UI Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .stButton>button { background-color: #28a745; color: white; font-size: 1.2rem; width: 100%; }
    .risk-predicted { color: #dc3545; font-weight: bold; }
    .no-risk { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Prediction Function ---
def predict_health_impact(input_df, model, mlb, feature_names):
    """Makes a prediction based on user input using the full pipeline."""
    input_aligned = pd.DataFrame(columns=feature_names)
    input_aligned = pd.concat([input_aligned, input_df], axis=0).fillna(0)
    input_aligned = input_aligned[feature_names]
    
    prediction_bin = model.predict(input_aligned)
    predicted_labels = mlb.inverse_transform(prediction_bin)
    return predicted_labels[0], input_aligned


# --- App Layout ---
st.markdown("<h1 class='main-header'>🧒 Kids Screen Time Health Impact Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Model Selection ---
st.sidebar.title("⚙️ Model Configuration")
selected_model_type = st.sidebar.selectbox(
    "Choose a Prediction Model",
    options=['ensemble', 'catboost', 'lightgbm', 'xgboost'],
    index=0
)

model, mlb, feature_names, explainers = load_artifacts(selected_model_type)

if model is None:
    st.error(f"❗ **Model artifacts for '{selected_model_type}' not found.** Please run `python main.py --model-type {selected_model_type}`.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("👤 Child's Information")
        age = st.slider("Age", 5, 18, 12)
        gender = st.selectbox("Gender", ["Male", "Female"])
        urban_or_rural = st.selectbox("Area", ["Urban", "Rural"])

        st.header("📱 Screen Time Habits")
        avg_daily_screen_time = st.slider("Average Daily Screen Time (hours)", 0.5, 10.0, 4.0, 0.1)
        primary_device = st.selectbox("Primary Device", ["Smartphone", "Laptop", "TV", "Tablet"])
        exceeded_limit = st.checkbox("Exceeded Recommended Screen Time Limit?", value=True)
        edu_to_rec_ratio = st.slider("Educational to Recreational Ratio", 0.0, 1.0, 0.4, 0.01)

    input_data = {
        'Age': age, 'Avg_Daily_Screen_Time_hr': avg_daily_screen_time,
        'Exceeded_Recommended_Limit': 1 if exceeded_limit else 0,
        'Educational_to_Recreational_Ratio': edu_to_rec_ratio,
        'Age_Screen_Time_Interaction': age * avg_daily_screen_time,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Urban_or_Rural_Urban': 1 if urban_or_rural == "Urban" else 0,
        'Primary_Device_Smartphone': 1 if primary_device == "Smartphone" else 0,
        'Primary_Device_TV': 1 if primary_device == "TV" else 0,
        'Primary_Device_Tablet': 1 if primary_device == "Tablet" else 0,
        'Screen_Time_Risk_Low': 1 if 2 < avg_daily_screen_time <= 4 else 0,
        'Screen_Time_Risk_Moderate': 1 if 4 < avg_daily_screen_time <= 6 else 0,
        'Screen_Time_Risk_High': 1 if 6 < avg_daily_screen_time <= 8 else 0,
        'Screen_Time_Risk_Very High': 1 if avg_daily_screen_time > 8 else 0,
    }
    input_df = pd.DataFrame([input_data])

    with col2:
        st.header("📈 Prediction Results")
        if st.button("Predict Health Impact"):
            predicted_impacts, input_aligned = predict_health_impact(input_df, model, mlb, feature_names)
            
            st.subheader(f"Predicted Health Impacts (using {selected_model_type.upper()}):")
            if predicted_impacts:
                for impact in predicted_impacts:
                    st.markdown(f" - <span class='risk-predicted'>⚠️ {impact}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='no-risk'>✅ No specific health impacts predicted.</p>", unsafe_allow_html=True)
            st.info("💡 **Disclaimer:** This is an AI-powered prediction and not medical advice.", icon="ℹ️")

            st.markdown("---")
            st.header("🔍 Prediction Insights (Why?)")

            if explainers:
                scaler = model.named_steps['scaler']
                input_scaled = scaler.transform(input_aligned)
                input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
                
                # --- CHANGED: New logic to handle both risky and safe predictions ---
                if predicted_impacts:
                    st.subheader("Why these risks were predicted:")
                    impacts_to_explain = predicted_impacts
                else:
                    st.subheader("Why no specific risks were predicted:")
                    st.write("The plots below show the factors that kept the risk low for each potential health impact.")
                    impacts_to_explain = mlb.classes_ # Explain all possible classes

                for model_name, explainer_list in explainers.items():
                    st.subheader(f"Insights from {model_name}:")
                    for impact in impacts_to_explain:
                        try:
                            class_index = list(mlb.classes_).index(impact)
                            st.write(f"Factors for '{impact}'")
                            
                            specific_explainer = explainer_list[class_index]
                            shap_values_for_class = specific_explainer.shap_values(input_scaled_df)
                            
                            fig = plt.figure()
                            shap.waterfall_plot(
                                shap.Explanation(
                                    values=shap_values_for_class[0], 
                                    base_values=specific_explainer.expected_value, 
                                    data=input_aligned.iloc[0],
                                    feature_names=feature_names
                                ), 
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(fig, bbox_inches='tight', clear_figure=True)
                            
                        except Exception as e:
                            st.error(f"Could not generate waterfall plot for {impact} using {model_name}: {e}")
            else:
                st.warning("Could not load explainers.")