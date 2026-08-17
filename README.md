🏦 Bank Term Deposit Prediction App

A complete machine learning pipeline and interactive Streamlit application designed to predict whether a bank client will subscribe to a term deposit. This project is part of the BITS Pilani WILP – M.Tech AI & ML coursework and demonstrates end to end ML development, model comparison, deployment, and user friendly interface design.

📘 1. Introduction

Financial institutions frequently run marketing campaigns to encourage clients to subscribe to term deposits. Predicting which clients are likely to subscribe helps optimize marketing efforts, reduce costs, and improve conversion rates.

This project uses the Bank Marketing Dataset (UCI) to build multiple machine learning models and deploys a Streamlit web application that allows users to:

•	Enter client details manually

•	Upload CSV files for batch predictions

•	Generate random client profiles

•	Select different ML models

•	View prediction results instantly

The repository contains all training scripts, model files, metrics, and the Streamlit app required for full reproducibility.

📊 2. Dataset Overview

Dataset: Bank Marketing Dataset (UCI Machine Learning Repository) 

Target Variable: y

•	yes → client subscribed

•	no → client did not subscribe

•	Converted to numeric: 1 and 0

Features Included

•	Demographic: age, job, marital, education

•	Financial: balance, housing loan, personal loan

•	Campaign-related: contact type, duration, campaign count, previous attempts

•	Outcome-related: poutcome (previous campaign result)

•	Temporal: month, day

Imbalance Handling

The dataset is highly imbalanced (majority class = “no”).

To address this, SMOTE oversampling is applied during training.

3. Machine Learning Models Implemented

Six classification models were trained and evaluated:

1.	Logistic Regression

2.	Decision Tree

3.	Random Forest

4.	k-Nearest Neighbors (kNN)

5.	Naive Bayes

6.	XGBoost

Each model is saved as a. joblib file inside the model/ directory for use in the Streamlit app.

4. Training Pipeline (ML_assignment_2.ipynb)

The training script performs:

Data Preprocessing

•	Load CSV

•	Clean column names

•	Convert target to numeric

•	One-hot encode categorical features

Train-Test Split

•	80% training

•	20% testing

•	Stratified split to preserve class distribution

SMOTE Oversampling

•	Applied only on training data

•	Balances minority class

Feature Scaling

•	StandardScaler applied to numeric features

•	Required for kNN and improves convergence for Logistic Regression

Model Training

Each model is trained, evaluated, and saved:

•	.joblib files stored in /model

•	Metrics stored in model_metrics.csv

Outputs Generated

•	model_metrics.csv

•	test_data.csv

•	/model/*.joblib files

5. Repository Structure

bank-term-deposit-app/

│

├── app.py                     # Streamlit application

├── ML_assignment_2.ipynb      # Model training script

├── requirements.txt           # Dependencies for Streamlit Cloud

├── model_metrics.csv          # Precision, Recall, F1-score, Accuracy, AUC, MCC

├── test_data.csv              # Test dataset for UI defaults

│

└── model/                     # Trained ML models
     
      ├── logistic_regression.joblib
    
      ├── decision_tree.joblib
     
      ├── knn.joblib
     
      ├── naive_bayes.joblib
    
      ├── random_forest.joblib
    
      └── xgboost.joblib

🎨 6. Streamlit App Features (app.py)

The Streamlit app provides an intuitive interface with:

Manual Input Form

Users can enter client details such as age, job, marital status, balance, loan status, etc.

Model Selection

Dropdown to choose any of the six trained models.

Random Client Generator

Creates a random client profile using real test data.

CSV Upload

Allows batch predictions for multiple clients.

Prediction Output

Displays:

•	YES → client likely to subscribe

•	NO → client unlikely to subscribe

🚀 7. Running the Application Locally

Step 1 — Install Dependencies

pip install -r requirements.txt

Step 2 — Run Streamlit App

streamlit run app.py

The app will open in your browser at:

http://localhost:8501

📈 8. Model Comparison (Based on Actual Results)

Below is the final comparison table generated from your training script:

Model	                Accuracy	AUC	    Precision	  Recall	  F1	    MCC

Logistic Regression	  0.9021	  0.9017	0.6393	    0.3752	  0.4729	0.4413

Decision Tree	        0.8735	  0.7041	0.4612	    0.4830	  0.4718	0.4002

Random Forest	        0.9071	  0.9281	0.6469	    0.4537	  0.5333	0.4929

kNN	                  0.8928	  0.8122	0.5829	    0.2958	  0.3925	0.3642

Naive Bayes	          0.8418	  0.7414	0.3580	    0.4442	  0.3965	0.3090

XGBoost	              0.9085	  0.9351	0.6260	    0.5425	  0.5813	0.5319

Professional Interpretation

XGBoost — Best Overall Model

Highest AUC (0.9351)

Highest MCC (0.5319)

Best F1 score

Strong recall → identifies more positive cases

Excellent balance across all metrics

Random Forest — Strong Runner-Up

High accuracy

High AUC

Good precision and recall balance

Logistic Regression — Strong Baseline

High AUC

Good precision

Lower recall → misses more positive cases


Decision Tree

Moderate performance

Lower AUC → weaker separation

kNN

Struggles with high-dimensional one-hot encoded data

Low recall

Naive Bayes

Weakest performance

Assumptions do not fit dataset characteristics

Conclusion

XGBoost is the best-performing model for predicting term deposit subscription due to its superior AUC, MCC, and F1 score.
Random Forest also performs strongly, while Logistic Regression remains a reliable baseline.
Naive Bayes and kNN are less suitable for this dataset.

📝 9. Assignment Deliverables Included

✔ ML_assignment_2 script

✔ Streamlit app

✔ Trained models

✔ Metrics CSV

✔ Test data

✔ Requirements file

✔ README.md
