import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("credit_data.csv")

print("Data loaded successfully")
print(df.head())

# -----------------------------
# 2. BASIC DATA VALIDATION
# -----------------------------
required_columns = [
    "age",
    "monthly_income",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "num_late_payments",
    "default"
]

missing_cols = set(required_columns) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

if df.isnull().sum().any():
    raise ValueError("Dataset contains missing values")

print("Data validation passed")

# -----------------------------
# 3. FEATURE / TARGET SPLIT
# -----------------------------
X = df.drop("default", axis=1)
y = df["default"]

# -----------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. FEATURE SCALING
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. MODEL TRAINING
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model training completed")

# -----------------------------
# 7. MODEL EVALUATION
# -----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\nFeature importance:")
print(feature_importance)
