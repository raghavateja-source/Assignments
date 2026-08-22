import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42
TARGET_COL = "grid_failure_flag"

DATA_PATH = "data/raw/PowerGrid_Utility_Intelligence.xlsx"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs/model_results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Load Data
# =========================================================

print("Loading dataset...")
df = pd.read_excel(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# =========================================================
# Remove duplicates
# =========================================================

before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"Removed {before - after} duplicate rows")

# =========================================================
# Validate target
# =========================================================

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

# Ensure binary integer target
df[TARGET_COL] = df[TARGET_COL].astype(int)

print(df[TARGET_COL].value_counts(normalize=True))

# =========================================================
# Drop leakage columns
# =========================================================

leakage_cols = [
    "estimated_revenue_loss",
    "regulatory_penalty_cost",
    "avg_outage_duration_minutes"
]

existing_leakage = [c for c in leakage_cols if c in df.columns]

print(f"Dropping leakage columns: {existing_leakage}")

df = df.drop(columns=existing_leakage)

# =========================================================
# Drop obvious identifier columns
# =========================================================

id_cols = [
    "asset_id",
    "legacy_asset_code",
    "monitoring_batch_id",
    "administrative_reference"
]

existing_ids = [c for c in id_cols if c in df.columns]

print(f"Dropping ID columns: {existing_ids}")

df = df.drop(columns=existing_ids)

# =========================================================
# Split features and target
# =========================================================

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Identify column types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# =========================================================
# Preprocessing
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =========================================================
# Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# =========================================================
# Models
# =========================================================

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_SEED
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
}

results = []

best_model_name = None
best_pr_auc = -1
best_pipeline = None

# =========================================================
# Training & Evaluation
# =========================================================

for name, model in models.items():

    print("=" * 60)
    print(f"Training: {name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    results.append({
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        best_model_name = name
        best_pipeline = pipeline

# =========================================================
# Save Results
# =========================================================

results_df = pd.DataFrame(results).sort_values(
    by="pr_auc",
    ascending=False
)

print("\\nModel Comparison")
print(results_df)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# Save best model
joblib.dump(
    best_pipeline,
    os.path.join(MODEL_DIR, "best_model.pkl")
)

# Save preprocessor separately
joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")
)

print("\\nBest model:", best_model_name)
print(f"Best PR-AUC: {best_pr_auc:.4f}")

# =========================================================
# Feature Importance (Tree Models)
# =========================================================

if best_model_name in ["random_forest", "decision_tree"]:

    model = best_pipeline.named_steps["model"]
    prep = best_pipeline.named_steps["preprocessor"]

    feature_names = prep.get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )

    print("\\nTop 20 Features")
    print(importance_df.head(20))

print("\\nTraining completed successfully.")