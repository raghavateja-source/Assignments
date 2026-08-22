"""Train and compare Logistic Regression, Decision Tree, Random Forest and SVM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from data_loader import load_data
from preprocessing import build_preprocessor
from config import *

ROOT = PROJECT_ROOT

def probability(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1/(1+np.exp(-np.clip(score, -30, 30)))

def main():
    df = load_data(ROOT / 'data/raw/original_dataset.xlsx').drop_duplicates().copy()
    X = df.drop(columns=[TARGET_COLUMN] + LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + BUSINESS_IMPACT_COLUMNS)
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=.50, stratify=y_tmp, random_state=RANDOM_STATE)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=30, min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE),
        'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    }
    rows=[]; fitted={}
    for name, estimator in models.items():
        pipe = Pipeline([('preprocessor', build_preprocessor(X_train)), ('model', estimator)])
        pipe.fit(X_train, y_train)
        p = probability(pipe, X_val)
        thresholds=np.linspace(.10,.90,161)
        best_t=max(thresholds, key=lambda t: f1_score(y_val, (p>=t).astype(int)))
        pred=(p>=best_t).astype(int)
        rows.append({'Model':name,'Precision':precision_score(y_val,pred),'Recall':recall_score(y_val,pred),'F1':f1_score(y_val,pred),'ROC-AUC':roc_auc_score(y_val,p),'PR-AUC':average_precision_score(y_val,p),'Threshold':best_t})
        fitted[name]=pipe
    results=pd.DataFrame(rows).sort_values(['PR-AUC','ROC-AUC','Recall'], ascending=False)
    results.to_csv(ROOT/'outputs/model_results/model_comparison.csv',index=False)
    best=results.iloc[0]['Model']; threshold=float(results.iloc[0]['Threshold'])
    final=Pipeline([('preprocessor',build_preprocessor(X)),('model':models[best])])
    final.fit(X,y)
    joblib.dump(final, ROOT/'models/final_model.pkl')
    joblib.dump(final.named_steps['preprocessor'], ROOT/'models/preprocessing_pipeline.pkl')
    json.dump({'threshold':threshold}, open(ROOT/'models/decision_threshold.json','w'), indent=2)
    print(results.to_string(index=False)); print('Selected:',best,'threshold:',threshold)

if __name__=='__main__': main()
