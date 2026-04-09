# backend/session_store.py
# ─────────────────────────────────────────────────────────────
# Manages the list of saved training runs in session state.
# v2: save_run() now accepts an `artifacts` dict that stores
#     pre-rendered plot PNG bytes and autotune cv_results,
#     keyed by artifact type. Page 6 reads these to build
#     per-run content selections for the report.
#
# Artifact keys (all optional):
#   "actual_vs_predicted"  : PNG bytes
#   "residuals"            : PNG bytes
#   "feature_importance"   : PNG bytes
#   "autotune_history"     : PNG bytes  (trial R² chart)
#   "autotune_metrics"     : dict       {best_params, best_score, method}
#   "autotune_comparison"  : list[dict] [{Metric, Original, Tuned, Δ}]
# ─────────────────────────────────────────────────────────────

import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config.model_registry import MODEL_REGISTRY


# ── Save / load ────────────────────────────────────────────

def save_run(
    model_key:     str,
    metrics:       dict,
    cfg_snapshot:  dict,
    feature_names: list[str],
    y_test:        np.ndarray,
    y_pred:        np.ndarray,
    best_params:   dict | None = None,
    dataset_name:  str  | None = None,
    target_col:    str  | None = None,
    input_features: list[str] | None = None,
    notes:         str         = "",
    artifacts:     dict | None = None,
) -> int:
    """
    Append a training run to st.session_state["runs"].

    Parameters
    ----------
    artifacts : dict of pre-rendered content to store with the run.
        Keys are artifact type strings; values are PNG bytes (plots)
        or plain dicts/lists (metrics tables). Only the keys chosen
        by the user on Page 1's smart-save widget are present.

        Supported keys:
          "actual_vs_predicted"  — PNG bytes
          "residuals"            — PNG bytes
          "feature_importance"   — PNG bytes
          "autotune_history"     — PNG bytes
          "autotune_metrics"     — dict  {best_params, best_score, method}
          "autotune_comparison"  — list[dict]

    Returns
    -------
    run_id : int (1-based)
    """
    if "runs" not in st.session_state:
        st.session_state["runs"] = []

    run = {
        "run_id":        len(st.session_state["runs"]) + 1,
        "timestamp":     datetime.datetime.now().strftime("%H:%M:%S"),
        "model_key":     model_key,
        "model_label":   MODEL_REGISTRY.get(model_key, {}).get("label", model_key),
        "metrics":       metrics,
        "best_params":   best_params or {},
        "cfg_snapshot":  cfg_snapshot,
        "feature_names": feature_names,
        "y_test":        y_test,
        "y_pred":        y_pred,
        "dataset_name":  dataset_name or st.session_state.get("active_dataset", "—"),
        "target_col":    target_col   or st.session_state.get("target_col",    "—"),
        "input_features": input_features or st.session_state.get("input_features", []),
        "notes":         notes,
        "artifacts":     artifacts or {},
    }

    st.session_state["runs"].append(run)
    return run["run_id"]


def get_runs() -> list[dict]:
    return st.session_state.get("runs", [])


def clear_runs() -> None:
    st.session_state["runs"] = []


def delete_run(run_id: int) -> None:
    runs = get_runs()
    st.session_state["runs"] = [r for r in runs if r["run_id"] != run_id]


# ── Artifact availability helpers ──────────────────────────

# Human-readable labels for each artifact key
ARTIFACT_LABELS: dict[str, str] = {
    "actual_vs_predicted": "Actual vs Predicted plot",
    "residuals":           "Residual distribution plot",
    "feature_importance":  "Feature importance plot",
    "autotune_history":    "Auto-tune trial history chart",
    "autotune_metrics":    "Auto-tune best params & score",
    "autotune_comparison": "Auto-tune metrics comparison table",
}


def available_artifacts(run: dict) -> list[str]:
    """
    Return list of artifact keys present in a run dict.
    Used by Page 6 to build per-run content checkboxes.
    """
    return [k for k in ARTIFACT_LABELS if k in run.get("artifacts", {})]


def artifact_label(key: str) -> str:
    return ARTIFACT_LABELS.get(key, key)


# ── Comparison table ───────────────────────────────────────

def runs_to_dataframe(runs: list[dict] | None = None) -> pd.DataFrame:
    """Return a tidy DataFrame for the Page 5 comparison table."""
    if runs is None:
        runs = get_runs()
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        m = r.get("metrics",      {})
        c = r.get("cfg_snapshot", {})
        a = r.get("artifacts",    {})
        rows.append({
            "#":           r["run_id"],
            "time":        r["timestamp"],
            "model":       r["model_label"],
            "dataset":     r["dataset_name"],
            "target":      r["target_col"],
            "R²":          m.get("R2",   None),
            "RMSE":        m.get("RMSE", None),
            "MAE":         m.get("MAE",  None),
            "MSE":         m.get("MSE",  None),
            "train_size":  c.get("train_size",     None),
            "split":       c.get("split_strategy", ""),
            "artifacts":   len(a),
            "notes":       r.get("notes", ""),
        })

    df = pd.DataFrame(rows)

    if len(df) > 1 and "R²" in df.columns:
        baseline_r2    = df["R²"].iloc[0]
        df["ΔR² vs #1"] = (df["R²"] - baseline_r2).round(5)

    return df


def get_best_run(metric: str = "R2") -> dict | None:
    runs = get_runs()
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("metrics", {}).get(metric, -np.inf))


# ── Plot helpers ───────────────────────────────────────────

def figure_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Serialise a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def figure_to_buffer(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    """Return a seeked BytesIO of a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf
