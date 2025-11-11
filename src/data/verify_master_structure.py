# src/data/verify_master_structure.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "verify_master_report.csv"

MASTER = INTERIM / "wgi_econ_master.csv"
if not MASTER.exists():
    raise SystemExit("Run the builder + clean_master first: missing " + str(MASTER))

df = pd.read_csv(MASTER, low_memory=False)

# basic info
rows = len(df)
cols = len(df.columns)
years = (int(df['year'].min()), int(df['year'].max())) if 'year' in df.columns else ('n/a','n/a')
n_countries = int(df['iso3'].nunique()) if 'iso3' in df.columns else 'n/a'

# missingness
missing = df.isna().mean().sort_values(ascending=False)
missing_df = missing.reset_index()
missing_df.columns = ['column','missing_fraction']

# coverage by year (counts)
year_counts = df.groupby('year')['iso3'].nunique().reset_index().rename(columns={'iso3':'n_countries'})

# coverage by country (years available)
country_years = df.groupby('iso3')['year'].agg(['min','max','count']).reset_index().rename(columns={'count':'n_obs'})

# value stats and suspicious scales
num = df.select_dtypes(include=['number'])
stats = num.describe().T[['count','mean','std','min','25%','50%','75%','max']].reset_index().rename(columns={'index':'column'})

# detect suspicious columns: extremely large means or huge max/min differences
suspicious = []
for c in stats['column']:
    r = stats[stats['column']==c].iloc[0]
    if (abs(r['mean'])>1e9) or (abs(r['max'])/ (abs(r['50%'])+1e-9) > 1e6):
        suspicious.append(c)

# write reports
INTERIM.mkdir(parents=True, exist_ok=True)
missing_df.to_csv(INTERIM / "verify_missingness.csv", index=False)
year_counts.to_csv(INTERIM / "verify_year_coverage.csv", index=False)
country_years.to_csv(INTERIM / "verify_country_coverage.csv", index=False)
stats.to_csv(INTERIM / "verify_value_stats.csv", index=False)

print("MASTER:", MASTER)
print("rows:", rows, "cols:", cols)
print("years:", years)
print("unique iso3:", n_countries)
print()
print("Top 12 most-missing columns:")
print(missing.head(12).to_string())
print()
print("Suspicious numeric columns (very large scale or outliers):")
print(suspicious)
print()
print("Saved reports to:", INTERIM)
