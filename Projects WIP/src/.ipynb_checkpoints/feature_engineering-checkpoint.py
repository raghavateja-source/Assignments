import pandas as pd

def engineer_features(df):

    if 'load_utilization_pct' in df.columns:
        df['high_load_flag'] = (df['load_utilization_pct'] > 85).astype(int)

    if 'equipment_health_score' in df.columns:
        df['poor_health_flag'] = (df['equipment_health_score'] < 50).astype(int)

    return df