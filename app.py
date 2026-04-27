import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sleep Disorder CSV Prediction", page_icon="😴", layout="centered")

st.title("😴 Sleep Disorder Prediction App (CSV Upload)")
st.write("Upload a CSV file to predict the Sleep Disorder Type for each person.")

# Load model
model = joblib.load('sleep_disorder_model.pkl')
class_label_map = {0: 'None', 1: 'Insomnia', 2: 'Sleep Apnea'}

def preprocess_input(df):
    df = df.copy()
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'BMI Category': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
        'Blood Pressure': {'Low': 0, 'Normal': 1, 'High': 2},
        'Smoking': {'No': 0, 'Yes': 1},
        'Alcohol Consumption': {'No': 0, 'Yes': 1},
        'Occupation': {'Doctor': 0, 'Engineer': 1, 'Teacher': 2, 'Nurse': 3, 'Lawyer': 4,
                       'Accountant': 5, 'Sales Representative': 6, 'Scientist': 7, 'Software Engineer': 8,
                       'Civil Engineer': 9, 'Pilot': 10, 'Police Officer': 11, 'Other': 12}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

uploaded_file = st.file_uploader("Upload CSV file here:", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Data:")
        st.write(df.head())

        # Preprocess
        processed_df = preprocess_input(df)

        # Drop unnecessary columns (Person ID, Sleep Disorder) + model-ignored columns
        processed_df = processed_df.drop(['Person ID', 'Sleep Disorder', 'Smoking', 'Alcohol Consumption'], axis=1, errors='ignore')

        # Ensure columns match what model expects
        model_features = model.feature_names_in_
        processed_df = processed_df[model_features]

        # Prediction
        predictions = model.predict(processed_df)
        readable_predictions = [class_label_map[pred] for pred in predictions]

        # Show result
        df['Predicted Sleep Disorder'] = readable_predictions
        st.subheader("Prediction Results with Sleep Disorder Type:")
        st.write(df)

        # Optional: plot distribution of predicted types
        st.subheader("Prediction Distribution:")
        result_counts = pd.Series(readable_predictions).value_counts()
        st.bar_chart(result_counts)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload a valid CSV file to see predictions.")

