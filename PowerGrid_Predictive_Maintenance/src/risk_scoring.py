import numpy as np
import pandas as pd

def add_risk_scores(df, probability_col='failure_probability'):
    out = df.copy()
    p = out[probability_col]
    out['risk_category'] = pd.cut(
        p, bins=[-np.inf, .30, .60, .80, np.inf],
        labels=['Low', 'Medium', 'High', 'Critical'], right=False
    )
    if 'customers_served' in out:
        cs = out['customers_served'].fillna(out['customers_served'].median())
        denom = cs.max() - cs.min()
        impact = (cs - cs.min()) / denom if denom else 0
        out['business_impact_index'] = impact
        out['maintenance_priority_score'] = p * (0.5 + 0.5 * impact)
    else:
        out['business_impact_index'] = np.nan
        out['maintenance_priority_score'] = p
    out['priority_rank'] = out['maintenance_priority_score'].rank(method='first', ascending=False).astype(int)
    out['recommended_action'] = np.select(
        [out['risk_category'].astype(str).eq('Critical'),
         out['risk_category'].astype(str).eq('High'),
         out['risk_category'].astype(str).eq('Medium')],
        ['Immediate engineering review / inspection',
         'Schedule preventive maintenance within 7 days',
         'Schedule condition inspection / monitor'],
        default='Routine monitoring'
    )
    return out
