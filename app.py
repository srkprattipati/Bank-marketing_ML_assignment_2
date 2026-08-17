import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load models
# -----------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
    "Decision Tree": joblib.load("model/decision_tree.joblib"),
    "Random Forest": joblib.load("model/random_forest.joblib"),
    "kNN": joblib.load("model/knn.joblib"),
    "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
    "XGBoost": joblib.load("model/xgboost.joblib")
}

# Load test data for random client generator
test_df = pd.read_csv("test_data.csv")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Bank Term Deposit Prediction App", layout="wide")
st.title("📊 Bank Term Deposit Prediction App")
st.write("Predict whether a client will subscribe to a term deposit.")

# -----------------------------
# Sidebar – Model Selection
# -----------------------------
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[selected_model]

# -----------------------------
# Helper: Preprocess input
# -----------------------------
def preprocess_input(df):
    # Load full dataset structure from training
    full_df = pd.read_csv("test_data.csv")
    full_df = full_df.drop("y", axis=1)

    # One-hot encode input
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=full_df.columns, fill_value=0)

    return df

# -----------------------------
# Manual Input Form
# -----------------------------
st.header("📝 Enter Client Details")

with st.form("client_form"):
    age = st.number_input("Age", min_value=18, max_value=95, value=30)
    job = st.selectbox("Job", test_df.filter(like="job_").columns.str.replace("job_", ""))
    marital = st.selectbox("Marital Status", test_df.filter(like="marital_").columns.str.replace("marital_", ""))
    education = st.selectbox("Education", test_df.filter(like="education_").columns.str.replace("education_", ""))
    balance = st.number_input("Balance", value=500)
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    contact = st.selectbox("Contact Type", test_df.filter(like="contact_").columns.str.replace("contact_", ""))
    month = st.selectbox("Month", test_df.filter(like="month_").columns.str.replace("month_", ""))
    poutcome = st.selectbox("Previous Outcome", test_df.filter(like="poutcome_").columns.str.replace("poutcome_", ""))

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "age": age,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        f"job_{job}": 1,
        f"marital_{marital}": 1,
        f"education_{education}": 1,
        f"contact_{contact}": 1,
        f"month_{month}": 1,
        f"poutcome_{poutcome}": 1
    }

    df_input = pd.DataFrame([input_dict])
    df_processed = preprocess_input(df_input)

    prediction = model.predict(df_processed)[0]
    result = "YES" if prediction == 1 else "NO"

    st.success(f"### Prediction: **{result}**")

# -----------------------------
# Random Client Generator
# -----------------------------
st.header("🎲 Generate Random Client")

if st.button("Generate Random Client"):
    random_client = test_df.sample(1).drop("y", axis=1)
    st.write(random_client)

    df_processed = preprocess_input(random_client)
    prediction = model.predict(df_processed)[0]
    result = "YES" if prediction == 1 else "NO"

    st.info(f"### Prediction: **{result}**")

# -----------------------------
# CSV Upload
# -----------------------------
st.header("📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    df_processed = preprocess_input(df_upload)

    preds = model.predict(df_processed)
    df_upload["Prediction"] = ["YES" if p == 1 else "NO" for p in preds]

    st.write(df_upload)
    st.download_button(
        label="Download Predictions",
        data=df_upload.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
