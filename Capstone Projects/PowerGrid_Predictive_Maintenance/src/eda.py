import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_data
from config import PROJECT_ROOT, TARGET_COLUMN

df=load_data(PROJECT_ROOT/'data/raw/original_dataset.xlsx')
out=PROJECT_ROOT/'outputs/figures'; out.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(5,4)); df[TARGET_COLUMN].value_counts().sort_index().plot(kind='bar'); plt.xticks([0,1],['No Failure','Failure'],rotation=0); plt.ylabel('Count'); plt.title('Grid Failure Distribution'); plt.tight_layout(); plt.savefig(out/'target_distribution.png',dpi=200); plt.close()
for col in ['asset_age_years','load_utilization_pct','equipment_health_score','maintenance_overdue_days','storm_risk_index']:
 if col in df:
  plt.figure(figsize=(6,4)); df.boxplot(column=col,by=TARGET_COLUMN); plt.title(f'{col} by Failure Flag'); plt.suptitle(''); plt.tight_layout(); plt.savefig(out/f'{col}_by_failure.png',dpi=200); plt.close()
