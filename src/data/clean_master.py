# src/data/clean_master.py
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
INTERIM = ROOT / "data" / "interim"
RAW_MASTER = INTERIM / "wgi_econ_master_raw.csv"
OUT_MASTER = INTERIM / "wgi_econ_master.csv"
OUT_UNION = INTERIM / "panel_union.csv"
OUT_CORE = INTERIM / "panel_core.csv"

if not RAW_MASTER.exists():
    raise SystemExit(f"Missing {RAW_MASTER}. Run the builder first.")

df = pd.read_csv(RAW_MASTER, low_memory=False)

# pick canonical country column
if "country_y" in df.columns:
    df["country"] = df["country_y"]
elif "country_x" in df.columns:
    df["country"] = df["country_x"]
elif "country" in df.columns:
    pass
else:
    # fallback: try to infer from other columns
    possible = [c for c in df.columns if c.lower().startswith("country")]
    if possible:
        df["country"] = df[possible[0]]
    else:
        df["country"] = None

# drop the old country_x / country_y if present
for c in ["country_x", "country_y"]:
    if c in df.columns:
        df = df.drop(columns=[c])

# ensure iso3 and year exist
if "iso3" not in df.columns:
    # try fallback column names
    for c in df.columns:
        if c.lower() in ("country code","iso","iso3","ref_area"):
            df = df.rename(columns={c:"iso3"})
            break

if "year" not in df.columns:
    for c in df.columns:
        if c.lower() in ("time_period","time","period"):
            df = df.rename(columns={c:"year"})
            break

# reorder columns: put country, iso3, year first
cols = list(df.columns)
rest = [c for c in cols if c not in ("country","iso3","year")]
newcols = ["country","iso3","year"] + rest
# keep only unique names in case any missing
newcols = [c for c in newcols if c in df.columns]
df = df[newcols]

# save cleaned master
df.to_csv(OUT_MASTER, index=False)
print(f"Saved cleaned master -> {OUT_MASTER} ({len(df)} rows, {len(df.columns)} cols)")

# write panel_union (same as master)
df.to_csv(OUT_UNION, index=False)
print(f"Saved panel union -> {OUT_UNION}")

# make panel_core: require iso3, year, gdp_usd and at least one WGI pillar not-null
wgi_pillars = [c for c in df.columns if c in ("voice_accountability","political_stability","gov_effectiveness","reg_quality","rule_of_law","control_corruption")]
required = ["iso3","year","gdp_usd"]
core = df.copy()
# drop rows missing iso3/year/gdp_usd
core = core.dropna(subset=[c for c in required if c in core.columns], how="any")
# keep only rows where at least one pillar present
if wgi_pillars:
    core = core[core[wgi_pillars].notna().any(axis=1)]
core.to_csv(OUT_CORE, index=False)
print(f"Saved panel core -> {OUT_CORE} ({len(core)} rows)")

# quick stats
print("\nQuick stats:")
print("Years:", df['year'].min() if 'year' in df.columns else 'n/a', "→", df['year'].max() if 'year' in df.columns else 'n/a')
print("Unique iso3:", df['iso3'].nunique() if 'iso3' in df.columns else 'n/a')
print("Columns:", len(df.columns))
print("Top 10 columns by missing fraction:")
missing = df.isna().mean().sort_values(ascending=False)
print(missing.head(10).to_string())
