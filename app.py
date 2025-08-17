import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Income Classifier", page_icon="💰", layout="centered")
st.title("💰 Income Classifier App")
st.caption("Predict whether income is >50K or <=50K using your trained pipeline")

# ---------- Load model (cached) ----------
@st.cache_resource
def load_pipeline():
    model_path = "income_pipeline.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "income_pipeline.pkl not found in the repo. Upload it to the repository root."
        )
    return joblib.load(model_path)

try:
    pipeline = load_pipeline()
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------- Input form ----------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=100, value=30)
        fnlwgt = st.number_input("Fnlwgt", min_value=1, max_value=2000000, value=200000)
        education = st.selectbox(
            "Education",
            [
                "Bachelors", "HS-grad", "Masters", "Some-college", "Assoc-voc",
                "Assoc-acdm", "Prof-school", "Doctorate", "7th-8th", "11th",
                "10th", "12th", "1st-4th", "5th-6th", "9th", "Preschool"
            ],
            index=0
        )
        education_num = st.number_input("Education Num", min_value=1, max_value=20, value=10)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40)

    with col2:
        workclass = st.selectbox(
            "Workclass",
            [
                "Private", "Self-emp-not-inc", "Self-emp-inc",
                "Federal-gov", "Local-gov", "State-gov",
                "Without-pay", "Never-worked"
            ]
        )
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Never-married", "Married-civ-spouse", "Divorced",
                "Separated", "Widowed", "Married-spouse-absent"
            ]
        )
        occupation = st.selectbox(
            "Occupation",
            [
                "Tech-support","Craft-repair","Other-service","Sales","Exec-managerial",
                "Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical",
                "Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv",
                "Armed-Forces"
            ]
        )
        relationship = st.selectbox(
            "Relationship",
            ["Not-in-family","Husband","Wife","Own-child","Unmarried","Other-relative"]
        )
        race = st.selectbox(
            "Race",
            ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"]
        )
        sex = st.selectbox("Sex", ["Male","Female"])
        native_country = st.text_input("Native Country", "United-States")

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    row = {
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
        "native_country": native_country
    }
    df = pd.DataFrame([row])

    try:
        pred = pipeline.predict(df)[0]
        label = ">50K" if int(pred) == 1 else "<=50K"
        st.subheader(f"Prediction: {label}")

        # If model supports predict_proba, show probability
        try:
            proba = float(pipeline.predict_proba(df)[0][1])
            st.write(f"Probability of >50K: **{proba:.2%}**")
        except Exception:
            pass

        st.caption("Tip: If you get odd results, ensure categories match training data.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
