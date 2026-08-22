import json
from pathlib import Path
import numpy as np

def probability_from_model(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))

def load_threshold(path):
    return json.loads(Path(path).read_text())['threshold']
