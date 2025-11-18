import pandas as pd
df = pd.read_csv("data/your_panel.csv")   # <- adjust path if needed

cols = ["exports_usd_ln_safe","imports_usd_ln_safe","fdi_inflow_usd_ln_safe","trade_exposure"]
print("=== pairwise correlations ===")
print(df[cols].corr().round(4))
print("\n=== variable summary ===")
print(df[cols].describe().T)
print("\n*** country obs counts (lowest 20) ***")
print(df.groupby("iso3")["year"].nunique().sort_values().head(20))

# extra quick checks that are helpful
print("\n*** count rows where exports==imports (after rounding) ***")
print((df["exports_usd_ln_safe"].round(6) == df["imports_usd_ln_safe"].round(6)).sum())

print("\n*** sample head for rows where any of the three trade vars is zero or identical ***")
mask = (df["exports_usd_ln_safe"].fillna(0)==0) | (df["imports_usd_ln_safe"].fillna(0)==0) | (df["fdi_inflow_usd_ln_safe"].fillna(0)==0)
print(df.loc[mask, ["iso3","year"] + cols].head(10))
