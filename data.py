# -*- coding: utf-8 -*-
"""Customer_Churn_Prediction_using_ML.ipynb

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 2. Load Data
df = pd.read_csv("/D/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"], inplace=True)

# Handle missing/empty values in 'TotalCharges'
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0").astype(float)

# Encode target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Label encode categorical features
object_columns = df.select_dtypes(include="object").columns
encoders = {}

for column in object_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# 3. Train-Test Split + SMOTE for class balancing
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

# 4. Model Training & Evaluation
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    print(f"{name} Accuracy: {np.mean(scores):.2f}")

# Train final model
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train_smote, y_train_smote)
y_pred = final_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save final model
model_data = {"model": final_model, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# 5. Load model and Predict on New Data
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# Sample input
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Encode input
input_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0]

print(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
print(f"Prediction Probability: {probability}")




