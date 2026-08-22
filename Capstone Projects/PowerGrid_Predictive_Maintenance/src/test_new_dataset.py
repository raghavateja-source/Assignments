"""Score a new PowerGrid asset dataset with the saved final model.

Usage:
    python src/test_new_dataset.py --input data/new/new_assets.xlsx
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from data_loader import load_data
from config import PROJECT_ROOT, TARGET_COLUMN, LEAKAGE_COLUMNS, IDENTIFIER_COLUMNS
from risk_scoring import add_risk_scores
from utils import load_threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV or Excel file to score')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'outputs/model_results/new_data_scored.csv'))
    args = parser.parse_args()
    df = load_data(args.input)
    model = joblib.load(PROJECT_ROOT / 'models/final_model.pkl')
    threshold = load_threshold(PROJECT_ROOT / 'models/decision_threshold.json')
    X = df.drop(columns=[TARGET_COLUMN] + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + ['customers_served'], errors='ignore')
    out = df.copy()
    out['failure_probability'] = model.predict_proba(X)[:, 1]
    out = add_risk_scores(out)
    out['model_alert_flag'] = (out['failure_probability'] >= threshold).astype(int)
    out = out.sort_values('maintenance_priority_score', ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'Scored {len(out):,} records -> {args.output}')

if __name__ == '__main__':
    main()
