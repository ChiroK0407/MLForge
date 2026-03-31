# backend/dataset_profiler.py
# ─────────────────────────────────────────────────────────────
# Generates a lightweight profile of any uploaded dataset.
# Called once after upload in Home.py.
# Results stored in st.session_state["dataset_profile"] and
# also embedded in the exported report.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy import stats


def profile_dataset(df: pd.DataFrame, target_col: str) -> dict:
    """
    Profile a dataframe and return a summary dict.

    Parameters
    ----------
    df         : raw uploaded dataframe
    target_col : name of the regression target column

    Returns
    -------
    dict with keys:
        n_rows, n_cols, n_features, missing_pct, is_time_series,
        feature_stats (DataFrame), target_stats (dict),
        warnings (list of str), dtypes_summary (dict)
    """

    warnings: list[str] = []

    # ── Basic counts ──────────────────────────────────────
    n_rows, n_cols = df.shape
    feature_cols   = [c for c in df.columns if c != target_col]
    n_features     = len(feature_cols)

    # ── Missing values ────────────────────────────────────
    total_cells  = n_rows * n_cols
    missing_vals = df.isnull().sum().sum()
    missing_pct  = 100 * missing_vals / total_cells if total_cells > 0 else 0.0

    if missing_pct > 20:
        warnings.append(
            f"⚠️ {missing_pct:.1f}% of values are missing. "
            "Consider cleaning before training."
        )
    elif missing_pct > 0:
        warnings.append(
            f"ℹ️ {missing_vals} missing values ({missing_pct:.1f}%). "
            "MLForge will impute with column means."
        )

    # ── Time-series detection ─────────────────────────────
    time_cols       = [c for c in df.columns if c.lower() in ("time", "t", "timestamp", "date")]
    is_time_series  = len(time_cols) > 0

    # ── Target column checks ──────────────────────────────
    if target_col in df.columns:
        target_series = df[target_col].dropna()

        # placeholder value check (e.g. IndPenSim uses -200)
        for placeholder in [-200, -999, -9999, 99999]:
            n_placeholder = (target_series == placeholder).sum()
            if n_placeholder > 0:
                warnings.append(
                    f"⚠️ Target column '{target_col}' contains {n_placeholder} "
                    f"placeholder values ({placeholder}). Clean before training."
                )

        # near-constant target
        cv = target_series.std() / (target_series.mean() + 1e-9)
        if abs(cv) < 0.01:
            warnings.append(
                f"⚠️ Target column '{target_col}' has very low variance (CV={cv:.4f}). "
                "Model may not learn meaningful patterns."
            )

        target_stats = {
            "mean":     float(target_series.mean()),
            "std":      float(target_series.std()),
            "min":      float(target_series.min()),
            "max":      float(target_series.max()),
            "skewness": float(stats.skew(target_series)),
            "kurtosis": float(stats.kurtosis(target_series)),
            "missing":  int(df[target_col].isnull().sum()),
        }

        if abs(target_stats["skewness"]) > 2:
            warnings.append(
                f"ℹ️ Target column is heavily skewed (skewness={target_stats['skewness']:.2f}). "
                "A log transform might improve model performance."
            )
    else:
        target_stats = {}

    # ── Per-feature statistics ────────────────────────────
    numeric_features = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for col in numeric_features:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "feature":    col,
            "dtype":      str(df[col].dtype),
            "missing":    int(df[col].isnull().sum()),
            "missing_%":  round(100 * df[col].isnull().sum() / n_rows, 1),
            "mean":       round(float(s.mean()), 4),
            "std":        round(float(s.std()), 4),
            "min":        round(float(s.min()), 4),
            "max":        round(float(s.max()), 4),
            "skewness":   round(float(stats.skew(s)), 3),
        })

    feature_stats = pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Dtype summary ─────────────────────────────────────
    dtypes_summary = {
        "numeric":  int((df.dtypes.apply(pd.api.types.is_numeric_dtype)).sum()),
        "datetime": int((df.dtypes.apply(pd.api.types.is_datetime64_any_dtype)).sum()),
        "object":   int((df.dtypes == "object").sum()),
    }

    non_numeric_features = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if non_numeric_features:
        warnings.append(
            f"ℹ️ Non-numeric columns will be dropped during preprocessing: "
            f"{non_numeric_features}"
        )

    return {
        "n_rows":         n_rows,
        "n_cols":         n_cols,
        "n_features":     n_features,
        "missing_pct":    round(missing_pct, 2),
        "missing_total":  int(missing_vals),
        "is_time_series": is_time_series,
        "time_cols":      time_cols,
        "feature_stats":  feature_stats,
        "target_stats":   target_stats,
        "dtypes_summary": dtypes_summary,
        "warnings":       warnings,
        "numeric_features": numeric_features,
    }
