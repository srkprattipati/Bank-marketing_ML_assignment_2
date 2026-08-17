import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Bank Term Deposit Prediction App")
st.write("Predict whether a client will subscribe to a term deposit.")

# Load models
model_file_map = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib",
    "XGBoost": "model/xgboost.joblib"
}

models = {name: joblib.load(path) for name, path in model_file_map.items()}

# Load test data
test_df = pd.read_csv("test_data.csv")
input_columns = list(test_df.drop("y", axis=1).columns)

st.header("Enter Client Details")

user_input = {}
for col in input_columns:
    if test_df[col].dtype != "object":
        user_input[col] = st.number_input(f"{col}", value=float(test_df[col].mean()))
    else:
        options = sorted(test_df[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)

input_df = pd.DataFrame([user_input])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=input_columns, fill_value=0)

st.write("### Input Data Preview")
st.dataframe(input_df)

model_choice = st.selectbox("Choose a model", list(models.keys()))

if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    result = "YES — Client will subscribe" if prediction == 1 else "NO — Client will not subscribe"
    st.success(result)

# CSV Upload
st.header("Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=input_columns, fill_value=0)

    model = models[model_choice]
    preds = model.predict(df_encoded)

    df["prediction"] = preds
    st.write(df)

# Random Client Generator
st.header("Generate Random Client")

if st.button("Generate Random Client"):
    random_client = {}

    for col in input_columns:
        if test_df[col].dtype != "object":
            low = test_df[col].min()
            high = test_df[col].max()
            random_client[col] = float(np.random.uniform(low, high))
        else:
            random_client[col] = np.random.choice(test_df[col].unique())

    random_df = pd.DataFrame([random_client])
    st.write(random_df)

    random_encoded = pd.get_dummies(random_df)
    random_encoded = random_encoded.reindex(columns=input_columns, fill_value=0)

    model = models[model_choice]
    pred = model.predict(random_encoded)[0]

    result = "YES" if pred == 1 else "NO"
    st.success(f"Prediction: {result}")
