#!/usr/bin/env python3
"""
Quick debug for reports/stage1_snapshot.json and reports/model_comparison_table.csv.

Run:
    python scripts/debug_snapshot_and_table.py --vars trade_exposure gov_index_zmean inflation_consumer_prices_pct
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import textwrap

ROOT = Path(".")
SNAP = ROOT / "reports" / "stage1_snapshot.json"
MODEL_TABLE = ROOT / "reports" / "model_comparison_table.csv"

def print_sep():
    print("="*80)

def show_snapshot(vars_list):
    if not SNAP.exists():
        print("SNAPSHOT MISSING:", SNAP)
        return
    raw = json.loads(SNAP.read_text(encoding="utf8"))
    print("SNAPSHOT PATH:", SNAP.resolve())
    print("snapshot type:", type(raw).__name__)
    if isinstance(raw, dict):
        keys = list(raw.keys())
        print("top-level keys (first 60):", keys[:60])
        # show a small sample from each requested var if present
        for v in vars_list:
            print_sep()
            print(f"VAR CHECK: {v}")
            if v in raw:
                print(f" - Found exact key '{v}'. Sample content (truncated):")
                s = raw[v]
                print(textwrap.indent(json.dumps(s, indent=2)[:1500], "  "))
            else:
                # try case-insensitive search through keys
                found = [k for k in keys if v.lower() in str(k).lower()]
                if found:
                    print(f" - Found keys matching substring: {found}")
                    for k in found:
                        print(f"   sample for key {k}:")
                        print(textwrap.indent(json.dumps(raw[k], indent=2)[:1500], "    "))
                else:
                    print(" - Not found as key; will try scanning nested dicts for matches...")
                    # scan nested dicts shallowly for varnames
                    matches = []
                    for k,val in raw.items():
                        try:
                            if isinstance(val, dict):
                                # look for var name inside nested dict keys or string values
                                if any(v.lower() in str(subk).lower() for subk in val.keys()):
                                    matches.append(k)
                                    break
                                # also search JSON dump for substring
                                if v.lower() in json.dumps(val).lower()[:2000]:
                                    matches.append(k)
                                    break
                        except Exception:
                            continue
                    if matches:
                        print(" - Possible parent keys containing var info:", matches)
                    else:
                        print(" - No matches in top-level dict or shallow nested content.")
    elif isinstance(raw, list):
        print("snapshot is a list of length:", len(raw))
        print("showing first 6 elements (truncated):")
        for i,el in enumerate(raw[:6]):
            print_sep()
            print(f"element {i} type: {type(el).__name__}")
            if isinstance(el, dict):
                print(" keys sample:", list(el.keys())[:20])
                # show snippet
                try:
                    print(textwrap.indent(json.dumps(el, indent=2)[:800], "  "))
                except Exception:
                    print("  (couldn't dump element)")
            else:
                print("  (non-dict element; type)", type(el))
        # try to find items matching vars_list
        for v in vars_list:
            print_sep()
            print("Searching list for var substring:", v)
            found_any = False
            for i,el in enumerate(raw):
                try:
                    s = json.dumps(el)[:1000].lower()
                    if v.lower() in s:
                        print(f" - found in list element index {i}; element keys (if dict):", list(el.keys())[:20] if isinstance(el,dict) else type(el))
                        found_any = True
                        break
                except Exception:
                    continue
            if not found_any:
                print(" - no match in first 1000 chars of list elements.")
    else:
        print("snapshot has unexpected top-level type:", type(raw).__name__)

def show_model_table(vars_list):
    if not MODEL_TABLE.exists():
        print("MODEL TABLE MISSING:", MODEL_TABLE)
        return
    df = pd.read_csv(MODEL_TABLE)
    print("MODEL TABLE PATH:", MODEL_TABLE.resolve())
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("head (5 rows):")
    print(df.head(5).to_string())
    # do a substring search for each var
    for v in vars_list:
        print_sep()
        print("Searching model_table for var substring:", v)
        mask = df["term"].astype(str).str.contains(v, case=False, na=False)
        print("matches found:", int(mask.sum()))
        if mask.any():
            print(df.loc[mask, ["model", "term", "coef", "std_err", "pvalue", "n_obs"]].head(20).to_string(index=False))
        else:
            # show first 30 terms for manual check
            print("No direct matches. First 30 'term' values for inspection:")
            print(df["term"].astype(str).head(30).to_string(index=False))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vars", nargs="+", required=True)
    args = p.parse_args()
    print_sep()
    show_snapshot(args.vars)
    print_sep()
    show_model_table(args.vars)
    print_sep()
    print("Done. Paste the full output here so I can generate the plotting run fix.")

if __name__ == "__main__":
    main()
