import pandas as pd

def engineer_features(df):
    """Create only defensible pre-failure features.

    The current production model intentionally uses the original measured
    variables after preprocessing; this function is provided as a controlled
    extension point and does not invent unsupported values.
    """
    out = df.copy()
    if 'load_utilization_pct' in out:
        out['high_load_flag'] = (out['load_utilization_pct'] >= 85).astype(int)
    if 'maintenance_overdue_days' in out:
        out['maintenance_overdue_flag'] = (out['maintenance_overdue_days'] > 0).astype(int)
    return out
