import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("income_pipeline.pkl")

st.title("💰 Income Classifier")
st.write("Predict whether income is >50K or <=50K")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
workclass = st.text_input("Workclass", "Private")
fnlwgt = st.number_input("Fnlwgt", min_value=0, value=100000)
education = st.text_input("Education", "Bachelors")
education_num = st.number_input("Education Num", min_value=1, value=10)
marital_status = st.text_input("Marital Status", "Never-married")
occupation = st.text_input("Occupation", "Tech-support")
relationship = st.text_input("Relationship", "Not-in-family")
race = st.text_input("Race", "White")
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.text_input("Native Country", "United-States")

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }])

    pred = pipeline.predict(input_data)[0]
    label = ">50K" if int(pred) == 1 else "<=50K"

    st.success(f"Predicted Income: **{label}**")
