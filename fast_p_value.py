
import pandas as pd
import numpy as np
from scipy import stats

def get_corr_p(s1, s2):
    mask = ~np.isnan(s1) & ~np.isnan(s2)
    if np.sum(mask) < 2: return np.nan, np.nan
    return stats.pearsonr(s1[mask], s2[mask])

filepath = r"c:\Users\egnan\OneDrive\Desktop\janu\backend\uploads\Sepsis data  5-Sheet 1-1-1-Sepsis excel sheetTable 1.csv"

try:
    df = pd.read_csv(filepath)
except:
    df = pd.read_csv(filepath, encoding='latin1')

df.columns = df.columns.str.strip()
col_map = {
    'Lactate': 'Lactate. 0hr',
    'Albumin': 'Albumin 0 hr',
    'CRP': 'CRP 0hr',
    'NLR': 'NLR 0hr',
    'PCT': 'PCT 0hr',
    'Outcome': 'Outcomes of the patient'
}

data = df[list(col_map.values())].copy()
data.columns = col_map.keys()

def clean_outcome(val):
    s = str(val).lower().strip()
    if 'expired' in s: return 1
    if 'relived' in s: return 0
    return np.nan

data['Outcome'] = data['Outcome'].apply(clean_outcome)
data = data.dropna(subset=['Outcome'])

def clean_numeric(val):
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except:
        return np.nan

feature_cols = ['Lactate', 'Albumin', 'CRP', 'NLR', 'PCT']
for col in feature_cols:
    data[col] = data[col].apply(clean_numeric)

data['LAR'] = data['Lactate'] / data['Albumin'].replace(0, np.nan)

# Calculate correlations and p-values
print("\n--- P-Value Results ---")
c, p = get_corr_p(data['LAR'], data['PCT'])
print(f"LAR vs PCT: Correlation={c:.4f}, P-Value={p:.4e}")

c, p = get_corr_p(data['LAR'], data['Outcome'])
print(f"LAR vs Outcome: Correlation={c:.4f}, P-Value={p:.4e}")

c, p = get_corr_p(data['PCT'], data['Outcome'])
print(f"PCT vs Outcome: Correlation={c:.4f}, P-Value={p:.4e}")
