# Model Evaluation Report

| Model               |   Precision |   Recall |       F1 |   ROC-AUC |   PR-AUC |   Threshold |
|:--------------------|------------:|---------:|---------:|----------:|---------:|------------:|
| Random Forest       |    0.740897 | 0.870428 | 0.800456 |  0.906451 | 0.881555 |       0.41  |
| Decision Tree       |    0.746914 | 0.862678 | 0.800633 |  0.90032  | 0.870022 |       0.4   |
| SVM                 |    0.649175 | 0.816801 | 0.723404 |  0.83067  | 0.790319 |       0.47  |
| Logistic Regression |    0.647475 | 0.822691 | 0.724642 |  0.83068  | 0.790316 |       0.415 |

Final selected model: **Random Forest**. Validation-selected threshold: **0.41**. Final holdout metrics are in `outputs/model_results/final_test_metrics.csv`.
