import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("transactions_data.csv")

print("Transaction data loaded")
print(df.head())

# -----------------------------
# 2. BASIC VALIDATION
# -----------------------------
required_columns = [
    "amount",
    "transaction_type",
    "merchant_category",
    "is_fraud"
]

missing_cols = set(required_columns) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

print("Validation passed")

# -----------------------------
# 3. ENCODE CATEGORICAL FEATURES
# -----------------------------
encoder = LabelEncoder()

df["transaction_type_encoded"] = encoder.fit_transform(df["transaction_type"])
df["merchant_category_encoded"] = encoder.fit_transform(df["merchant_category"])

# -----------------------------
# 4. FEATURE / TARGET SPLIT
# -----------------------------
X = df[
    ["amount", "transaction_type_encoded", "merchant_category_encoded"]
]
y = df["is_fraud"]

# -----------------------------
# 5. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 6. SIMPLE RULE-BASED CLASSIFIER
# (used for testing & validation)
# -----------------------------
def simple_classifier(row):
    if row["amount"] > 80000 and row["transaction_type_encoded"] == 2:
        return 1
    return 0

y_pred = X_test.apply(simple_classifier, axis=1)

# -----------------------------
# 7. EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nClassifier Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
