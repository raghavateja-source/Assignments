# PowerGrid Predictive Maintenance

## Business Objective
Predict `grid_failure_flag`, identify major failure drivers, and prioritize assets for proactive inspection and maintenance.

## Dataset
The supplied workbook contains 50,500 rows and 32 columns. Exact duplicate records: 500. Failure rate: 43.02%. No reliable date/time field was available.

## Leakage Controls
Excluded from predictive features: `estimated_revenue_loss`, `regulatory_penalty_cost`, and `avg_outage_duration_minutes`. `customers_served` is used separately for business-impact prioritization. Identifier-like fields are excluded from prediction.

## Models
Logistic Regression, Decision Tree, Random Forest, and linear SVM.

## Actual Model Comparison

| Model               |   Precision |   Recall |       F1 |   ROC-AUC |   PR-AUC |   Threshold |
|:--------------------|------------:|---------:|---------:|----------:|---------:|------------:|
| Random Forest       |    0.740897 | 0.870428 | 0.800456 |  0.906451 | 0.881555 |       0.41  |
| Decision Tree       |    0.746914 | 0.862678 | 0.800633 |  0.90032  | 0.870022 |       0.4   |
| SVM                 |    0.649175 | 0.816801 | 0.723404 |  0.83067  | 0.790319 |       0.47  |
| Logistic Regression |    0.647475 | 0.822691 | 0.724642 |  0.83068  | 0.790316 |       0.415 |

## Selected Model
**Random Forest** selected using validation PR-AUC/ROC-AUC and recall/F1. Final holdout metrics are stored in `outputs/model_results/final_test_metrics.csv`. The operational threshold selected on validation data is 0.41.

## Risk Framework
Low <30%, Medium 30–60%, High 60–80%, Critical >=80%. Maintenance priority combines failure probability with normalized customers served, kept separate from predictive modeling.

## How to Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/audit.py
python src/eda.py
python src/train.py
python src/test_new_dataset.py --input data/new/new_assets_template.csv
```

## Submission Contents
- `src/`: modular Python code
- `notebooks/`: student-friendly analysis notebooks
- `models/`: saved model and threshold
- `outputs/`: figures, tables, scored assets
- `reports/`: report, presentation, and supporting documentation

## Important Limitation
Because no reliable timestamp was present, temporal train/test validation could not be performed. The model should be recalibrated against real operational data before production use.
