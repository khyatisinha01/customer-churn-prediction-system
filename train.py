import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import pickle

# ── 1. Load & clean data ────────────────────────────────────────────────────
df = pd.read_csv("churn.csv")

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# TotalCharges has whitespace entries — coerce to NaN and drop
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean() * 100:.1f}%\n")

# ── 2. Encode target ────────────────────────────────────────────────────────
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── 3. One-hot encode categorical features ──────────────────────────────────
str_cols = [c for c in df.columns
            if str(df[c].dtype) in ("str", "string", "object") and c != "Churn"]
df = pd.get_dummies(df, columns=str_cols, drop_first=True)

# Force all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

X = df.drop("Churn", axis=1)
y = df["Churn"]

print(f"Features after encoding: {X.shape[1]}\n")

# ── 4. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Feature scaling (for Logistic Regression) ────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Benchmark 3 algorithms with 5-fold cross-validation ──────────────────
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_sc, X_test_sc),
    "Decision Tree":       (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model     = None
best_f1        = 0
best_model_name = ""

print("=" * 55)
print("Model Benchmarking Results")
print("=" * 55)

for name, (model, X_tr, X_te) in models.items():
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    print(f"\n{name}:")
    print(f"  CV Accuracy (5-fold) : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Test Accuracy        : {acc*100:.2f}%")
    print(f"  F1 Score             : {f1:.4f}")
    print(f"  Precision            : {prec:.4f}")
    print(f"  Recall               : {rec:.4f}")

    if f1 > best_f1:
        best_f1         = f1
        best_model      = model
        best_model_name = name

print("\n" + "=" * 55)
print(f"Best model: {best_model_name} (F1: {best_f1:.4f})")
print("=" * 55)

# ── 7. Full report for best model ───────────────────────────────────────────
if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_sc)
else:
    y_pred_best = best_model.predict(X_test)

print(f"\nClassification Report — {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# ── 8. Save best model ──────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nModel saved as model.pkl")
