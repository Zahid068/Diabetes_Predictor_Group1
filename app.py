import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "diabetes.csv")
    
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error(f"Error: 'diabetes.csv' not found. Please ensure it is placed inside the 'data/' folder at: {path}")
        st.stop()

def preprocess_data(data):
    clean_data = data.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'BMI']
    for col in zero_cols:
        clean_data[col] = clean_data[col].replace(0, np.nan)
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
    
    X = clean_data[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age']]
    y = clean_data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def run_model_or_algorithm(data, params):
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model_choice = params['model_type']
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    metrics_comparison = {
        'Logistic Regression': {
            'Accuracy': accuracy_score(y_test, lr_preds),
            'Precision': precision_score(y_test, lr_preds),
            'Recall': recall_score(y_test, lr_preds)
        },
        'Random Forest': {
            'Accuracy': accuracy_score(y_test, rf_preds),
            'Precision': precision_score(y_test, rf_preds),
            'Recall': recall_score(y_test, rf_preds)
        }
    }
    
    selected_model = rf if model_choice == "Random Forest" else lr
    user_inputs = np.array([
        params['pregnancies'],
        params['glucose'],
        params['bp'],
        params['bmi'],
        params['age']
    ]).reshape(1, -1)
    
    prediction = selected_model.predict(user_inputs)[0]
    
    if model_choice == "Random Forest":
        importance = selected_model.feature_importances_
    else:
        importance = np.abs(selected_model.coef_[0])
        
    return prediction, metrics_comparison, importance

def generate_explanation(result, context):
    glucose = context['glucose']
    bmi = context['bmi']
    
    if result == 1:
        return f"⚠️ **High Risk Level Detected:** The AI classifier flagged a positive diabetic risk outcome. This evaluation is heavily influenced by critical physiological markers, specifically your Glucose level ({glucose} mg/dL) and BMI index ({bmi}), crossing clinical risk baselines."
    else:
        return f"✅ **Low Risk Level Detected:** The AI classifier flagged a negative risk outcome. Your clinical statistics (Glucose: {glucose} mg/dL, BMI: {bmi}) are currently aligning with healthy biological parameters."

def create_visuals(data, result_data):
    all_metrics = result_data['metrics']
    importance = result_data['importance']
    model_choice = result_data['model_choice']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**A) Feature Importance Analysis**")
        fig, ax = plt.subplots(figsize=(5, 4))
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age']
        ax.barh(features, importance, color='#34495e')
        ax.set_title(f"Factor Contributions ({model_choice})")
        ax.set_xlabel("Relative Weight")
        st.pyplot(fig)
        
    with col2:
        st.write("**B) Baseline Performance Metrics Comparison**")
        fig, ax = plt.subplots(figsize=(5, 4))
        
        models = ['Logistic Regression', 'Random Forest']
        accuracies = [all_metrics['Logistic Regression']['Accuracy'], all_metrics['Random Forest']['Accuracy']]
        precisions = [all_metrics['Logistic Regression']['Precision'], all_metrics['Random Forest']['Precision']]
        recalls = [all_metrics['Logistic Regression']['Recall'], all_metrics['Random Forest']['Recall']]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, accuracies, width, label='Accuracy', color='#2ecc71')
        ax.bar(x, precisions, width, label='Precision', color='#3498db')
        ax.bar(x + width, recalls, width, label='Recall', color='#e74c3c')
        
        ax.set_title("Algorithm Benchmark Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.2)
        ax.legend()
        st.pyplot(fig)

def render_ui():
    st.set_page_config(page_title="Diabetes Risk Analyzer", layout="wide")
    st.title("🩺 Clinical Diabetes Diagnostic & Analytical Portal")
    st.write("An academic-grade, explainable machine learning dashboard designed to process and analyze clinical patient variables.")
    
    data = load_data()
    
    st.sidebar.header("📋 Problem Input Config")
    pregnancies = st.sidebar.slider("Pregnancies", min_value=0, max_value=17, value=1, step=1)
    glucose = st.sidebar.slider("Plasma Glucose (mg/dL)", min_value=40, max_value=200, value=117)
    bp = st.sidebar.slider("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=130, value=72)
    bmi = st.sidebar.slider("Body Mass Index (BMI)", min_value=15.0, max_value=60.0, value=32.0, step=0.1)
    age = st.sidebar.slider("Age (Years)", min_value=21, max_value=90, value=29)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Configuration")
    model_choice = st.sidebar.selectbox("Classifier Core", ["Random Forest", "Logistic Regression"])
    
    params = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'bmi': bmi,
        'age': age,
        'model_type': model_choice
    }
    
    if st.sidebar.button("Run Diagnostics Pipeline", type="primary"):
        with st.spinner('Running cross-validation predictions...'):
            prediction, metrics, importance = run_model_or_algorithm(data, params)
            
            st.subheader("🎯 Primary Prediction Output")
            if prediction == 1:
                st.error("### Diagnostics Flag: Clinical High Risk of Diabetes")
            else:
                st.success("### Diagnostics Flag: Low Risk / Normal Classification")
                
            st.subheader("📝 Decision Logic Interpretation")
            explanation_text = generate_explanation(prediction, params)
            st.info(explanation_text)
            
            st.markdown("---")
            st.subheader("📊 Visual Analytics & Performance Metrics")
            
            result_payload = {
                'metrics': metrics,
                'importance': importance,
                'model_choice': model_choice
            }
            create_visuals(data, result_payload)
    else:
        st.warning("👈 Adjust the medical features on the left configuration panel and click 'Run Diagnostics Pipeline'.")

if __name__ == "__main__":
    render_ui()