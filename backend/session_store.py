# backend/session_store.py
# ─────────────────────────────────────────────────────────────
# Manages the list of saved training runs in session state.
# Pages call save_run() after training; Page 5 reads get_runs().
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
    model_key:    str,
    metrics:      dict,
    cfg_snapshot: dict,
    feature_names: list[str],
    y_test:       np.ndarray,
    y_pred:       np.ndarray,
    best_params:  dict | None = None,
    dataset_name: str | None = None,
    target_col:   str | None = None,
    notes:        str = "",
) -> int:
    """
    Append a training run to st.session_state["runs"].

    Returns
    -------
    run_index : int (1-based)
    """
    if "runs" not in st.session_state:
        st.session_state["runs"] = []

    run = {
        "run_id":       len(st.session_state["runs"]) + 1,
        "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"),
        "model_key":    model_key,
        "model_label":  MODEL_REGISTRY.get(model_key, {}).get("label", model_key),
        "metrics":      metrics,
        "best_params":  best_params or {},
        "cfg_snapshot": cfg_snapshot,
        "feature_names": feature_names,
        "y_test":       y_test,
        "y_pred":       y_pred,
        "dataset_name": dataset_name or st.session_state.get("active_dataset", "—"),
        "target_col":   target_col   or st.session_state.get("target_col", "—"),
        "notes":        notes,
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


# ── Comparison table ───────────────────────────────────────

def runs_to_dataframe(runs: list[dict] | None = None) -> pd.DataFrame:
    """
    Return a tidy DataFrame of all saved runs for Page 5 comparison table.

    Columns: run_id, timestamp, model, dataset, target,
             R2, RMSE, MAE, MSE, train_size, split_strategy, notes
    """
    if runs is None:
        runs = get_runs()
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        m = r.get("metrics", {})
        c = r.get("cfg_snapshot", {})
        rows.append({
            "#":              r["run_id"],
            "time":           r["timestamp"],
            "model":          r["model_label"],
            "dataset":        r["dataset_name"],
            "target":         r["target_col"],
            "R²":             m.get("R2",   None),
            "RMSE":           m.get("RMSE", None),
            "MAE":            m.get("MAE",  None),
            "MSE":            m.get("MSE",  None),
            "train_size":     c.get("train_size", None),
            "split":          c.get("split_strategy", ""),
            "notes":          r.get("notes", ""),
        })

    df = pd.DataFrame(rows)

    # Delta columns vs first run
    if len(df) > 1 and "R²" in df.columns:
        baseline_r2 = df["R²"].iloc[0]
        df["ΔR² vs #1"] = (df["R²"] - baseline_r2).round(5)

    return df


def get_best_run(metric: str = "R2") -> dict | None:
    """Return the run with the highest value for `metric`."""
    runs = get_runs()
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("metrics", {}).get(metric, -np.inf))


# ── Plot helpers ───────────────────────────────────────────

def figure_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Serialize a matplotlib figure to PNG bytes (for report embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def figure_to_buffer(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    """Return a BytesIO buffer of a matplotlib figure (for st.image or docx)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf
