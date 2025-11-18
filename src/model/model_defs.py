# src/model/model_defs.py
"""
Model helpers: design matrix preparation, fixed-effects expansion, and lightweight checks.

This module keeps modeling utilities small and well-tested:
- prepare_design_matrix: builds numpy X and column labels (optionally adds constant)
- country_fixed_effects: append country dummy variables (drop_first option)
- safe_select_columns: helper that warns about missing predictors

These are intentionally simple, transparent helpers (no hidden magic) so results are
easy to explain in a methods section or SOP.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import logging

LOG = logging.getLogger("src.model.model_defs")
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def safe_select_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """Return list of columns from `cols` that exist in df; log missing ones.

    This makes configs tolerant to optional predictors.
    """
    cols = list(cols)
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        LOG.warning("Requested columns not found in dataframe (they will be skipped): %s", missing)
    return present


def prepare_design_matrix(df: pd.DataFrame, predictors: Iterable[str], add_constant: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Build a design matrix X (numpy array) and return (X, column_names).

    - predictors: list of column names expected in df (will be filtered by existence)
    - add_constant: if True, a leading constant column named 'const' is added

    Returns:
        X: numpy.ndarray, shape (n_obs, n_features)
        cols: list[str] column names matching columns in X (in order)
    """
    preds = safe_select_columns(df, list(predictors))
    if add_constant:
        # const first to match statsmodels convention
        cols = ["const"] + preds
        Xdf = pd.concat([pd.Series(1.0, index=df.index, name="const"), df[preds]], axis=1)
    else:
        cols = preds
        Xdf = df[preds].copy()

    # Coerce to numeric where possible; leave NaNs for caller to handle
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    X = Xdf.values
    LOG.info("Prepared design matrix with columns: %s (shape=%s)", cols, X.shape)
    return X, cols


def country_fixed_effects(df: pd.DataFrame, country_col: str = "iso3", drop_first: bool = True, prefix: str = "FE") -> pd.DataFrame:
    """
    Return a new DataFrame with country dummy variables appended.

    The function:
    - converts country codes to strings (safer if numeric iso codes exist),
    - creates one-hot encoded dummies with given prefix,
    - drops the original country column by default (keeps original if user wants it).

    Note: for large panels with many countries this explodes dimensionally; use sparingly.
    """
    if country_col not in df.columns:
        raise KeyError(f"country_col '{country_col}' not found in DataFrame")

    # ensure string values
    country_series = df[country_col].astype(str).fillna("NA")
    dummies = pd.get_dummies(country_series, prefix=prefix, drop_first=drop_first)
    # align index to original df
    dummies.index = df.index
    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    LOG.info("Appended %d country fixed-effect columns (prefix=%s).", dummies.shape[1], prefix)
    return out


def check_enough_data_for_regression(df: pd.DataFrame, required: Iterable[str]) -> bool:
    """
    Quick guard: returns True if after dropping NA rows for `required` there are >= 10 observations.
    This is a soft rule to avoid silent OLS on extremely small samples.
    """
    req = list(required)
    present = [c for c in req if c in df.columns]
    if not present:
        LOG.error("No required columns present in dataframe: %s", req)
        return False
    n = df.dropna(subset=present).shape[0]
    LOG.info("Observations with complete required columns (%s): %d", present, n)
    return n >= 10
