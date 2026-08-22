import os
import joblib
import numpy as np
import pandas as pd

# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/best_model.pkl"

# New dataset to score
NEW_DATA_PATH = "data/new/new_assets_to_score.xlsx"

# Output file
OUTPUT_PATH = "outputs/scored_new_assets.csv"

# Columns that were excluded during training
LEAKAGE_COLS = [
    "estimated_revenue_loss",
    "regulatory_penalty_cost",
    "avg_outage_duration_minutes"
]

ID_COLS = [
    "asset_id",
    "legacy_asset_code",
    "monitoring_batch_id",
    "administrative_reference"
]

TARGET_COL = "grid_failure_flag"

# =========================================================
# Load trained model
# =========================================================

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# =========================================================
# Load new dataset
# =========================================================

print("Loading new dataset...")

if NEW_DATA_PATH.endswith(".csv"):
    new_df = pd.read_csv(NEW_DATA_PATH)
else:
    new_df = pd.read_excel(NEW_DATA_PATH)

print(f"New dataset shape: {new_df.shape}")

# Keep original copy for output
output_df = new_df.copy()

# =========================================================
# Preserve asset_id if available
# =========================================================

if "asset_id" in new_df.columns:
    asset_ids = new_df["asset_id"]
else:
    asset_ids = pd.Series(range(1, len(new_df) + 1), name="asset_id")

# =========================================================
# Remove target column if present
# =========================================================

new_df = new_df.drop(columns=[TARGET_COL], errors="ignore")

# =========================================================
# Remove leakage columns
# =========================================================

new_df = new_df.drop(columns=LEAKAGE_COLS, errors="ignore")

# =========================================================
# Remove ID columns from features
# =========================================================

feature_df = new_df.drop(columns=ID_COLS, errors="ignore")

print(f"Feature shape for prediction: {feature_df.shape}")

# =========================================================
# Predict probabilities
# =========================================================

print("Generating predictions...")

failure_probability = model.predict_proba(feature_df)[:, 1]

# Optional binary prediction using 0.60 threshold
predicted_failure = (failure_probability >= 0.60).astype(int)

# =========================================================
# Risk category function
# =========================================================

def risk_category(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Critical"

risk_categories = [risk_category(p) for p in failure_probability]

# =========================================================
# Maintenance recommendation
# =========================================================

def maintenance_action(risk):
    if risk == "Critical":
        return "Immediate engineering inspection"
    elif risk == "High":
        return "Schedule preventive maintenance within 7 days"
    elif risk == "Medium":
        return "Inspect during next maintenance cycle"
    else:
        return "Routine monitoring"

recommended_actions = [maintenance_action(r) for r in risk_categories]

# =========================================================
# Build scored output
# =========================================================

scored_df = pd.DataFrame({
    "asset_id": asset_ids,
    "failure_probability": np.round(failure_probability, 4),
    "predicted_failure": predicted_failure,
    "risk_category": risk_categories,
    "recommended_action": recommended_actions
})

# Add useful business columns if available
useful_cols = [
    "asset_type",
    "substation_region",
    "customers_served",
    "equipment_health_score",
    "load_utilization_pct",
    "maintenance_overdue_days"
]

for col in useful_cols:
    if col in output_df.columns:
        scored_df[col] = output_df[col]

# Sort by highest risk
scored_df = scored_df.sort_values(
    by="failure_probability",
    ascending=False
).reset_index(drop=True)

# Add priority rank
scored_df.insert(0, "priority_rank", range(1, len(scored_df) + 1))

# =========================================================
# Save results
# =========================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

scored_df.to_csv(OUTPUT_PATH, index=False)

print("\\nScoring completed successfully.")
print(f"Results saved to: {OUTPUT_PATH}")

print("\\nTop 10 High-Risk Assets")
print(scored_df.head(10))