# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── 2. Preprocess ─────────────────────────────────────────────
# Drop customerID if it exists (column name varies by dataset source)
df.drop(columns=[c for c in df.columns if "customer" in c.lower()], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode all object columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Train with MLflow tracking ─────────────────────────────
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)

    # Log params & metrics to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc",  auc)
    mlflow.log_metric("f1_score", f1)

    # Log & register model
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn-model",
    )

    # Save locally for FastAPI to use
    os.makedirs("api", exist_ok=True)
    joblib.dump(model, "api/model.pkl")

    print(f"✅  Accuracy : {acc:.4f}")
    print(f"✅  ROC-AUC  : {auc:.4f}")
    print(f"✅  F1 Score : {f1:.4f}")
    print("✅  Model saved to api/model.pkl")