import pandas as pd
from config import PROJECT_ROOT, TARGET_COLUMN
from data_loader import load_data

path = PROJECT_ROOT / 'data/raw/original_dataset.xlsx'
df = load_data(path)
print('Shape:', df.shape)
print('\nColumns:')
print(df.columns.tolist())
print('\nData types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isna().sum().sort_values(ascending=False))
print('\nDuplicate rows:', df.duplicated().sum())
print('\nTarget distribution:')
print(df[TARGET_COLUMN].value_counts(dropna=False))
print('\nFailure rate:', df[TARGET_COLUMN].mean())
