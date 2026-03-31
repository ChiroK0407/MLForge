# backend/analyze_helper.py
# ─────────────────────────────────────────────────────────────
# AI-style model performance interpretation.
# Returns structured text instead of calling st.* directly,
# so the same analysis can be shown in the UI AND embedded
# in the exported report without duplication.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from config.model_registry import MODEL_REGISTRY

# ── R² quality thresholds ─────────────────────────────────
R2_THRESHOLDS = [
    (0.95, "excellent",  "badge-good",
     "Excellent fit --- model explains over 95% of variance in the test set."),
    (0.85, "strong",     "badge-good",
     "Strong fit. Minor residual variance; consider checking for overfitting."),
    (0.70, "moderate",   "badge-warn",
     "Moderate fit. Further feature engineering or tuning may improve results."),
    (0.50, "weak",       "badge-warn",
     "Weak predictive power. Review feature relevance or try a different model."),
    (0.00, "poor",       "badge-poor",
     "Poor fit. Model barely outperforms a constant baseline."),
    (-999, "failing",    "badge-poor",
     "Model performs worse than a constant baseline (R² < 0). "
     "Likely a data quality or feature mismatch issue."),
]


def interpret_r2(r2: float) -> tuple[str, str, str]:
    """
    Return (quality_label, badge_class, interpretation_text) for a given R².
    """
    for threshold, label, badge, text in R2_THRESHOLDS:
        if r2 >= threshold:
            return label, badge, text
    return "failing", "badge-poor", R2_THRESHOLDS[-1][3]


def interpret_metrics(
    metrics:    dict,
    model_key:  str,
    df,                 # pd.DataFrame from UI, or {} / None from report builder
    target_col: str,
) -> dict:
    """
    Produce a structured analysis dict from training metrics.

    Parameters
    ----------
    metrics    : {MSE, RMSE, MAE, R2}
    model_key  : key from MODEL_REGISTRY
    df         : raw dataframe (for data-quality checks).
                 Safely accepts an empty dict or None when called from
                 the report builder --- MAE/RMSE range context and
                 data-quality warnings are skipped in that case.
    target_col : regression target column

    Returns
    -------
    dict with keys:
        r2_label, r2_badge, r2_text,
        mae_context, rmse_context,
        data_warnings  (list of str),
        recommendations (list of str),
        summary_sentence (str --- used in report)
    """
    r2   = metrics.get("R2",   None)
    mae  = metrics.get("MAE",  None)
    rmse = metrics.get("RMSE", None)

    model_label = MODEL_REGISTRY.get(model_key, {}).get("label", model_key)

    # ── R² interpretation ─────────────────────────────────
    r2_label, r2_badge, r2_text = interpret_r2(r2 if r2 is not None else -999)

    # ── MAE / RMSE context ────────────────────────────────
    # Guard: df may be an empty dict {} when called from report_builder.py
    mae_context  = ""
    rmse_context = ""

    if isinstance(df, pd.DataFrame) and target_col in df.columns and mae is not None:
        target_range = float(df[target_col].max() - df[target_col].min())
        if target_range > 0:
            mae_pct  = 100 * mae  / target_range
            rmse_pct = 100 * rmse / target_range if rmse else 0
            mae_context  = (
                f"MAE of {mae:.4f} represents {mae_pct:.1f}% of the target range "
                f"({df[target_col].min():.3f} -- {df[target_col].max():.3f})."
            )
            rmse_context = (
                f"RMSE of {rmse:.4f} is {rmse_pct:.1f}% of the target range --- "
                f"{'acceptable' if rmse_pct < 10 else 'elevated, review outliers'}."
            )

    # ── Data quality warnings ─────────────────────────────
    # Guard: df may be an empty dict {} when called from report_builder.py
    data_warnings = []

    if isinstance(df, pd.DataFrame) and target_col in df.columns:
        for placeholder in [-200, -999, -9999]:
            n = (df[target_col] == placeholder).sum()
            if n > 0:
                data_warnings.append(
                    f"Target column contains {n} placeholder values ({placeholder})."
                )

        missing_n = df[target_col].isna().sum()
        if missing_n > 0:
            data_warnings.append(
                f"Target column has {missing_n} missing values (imputed with mean)."
            )

    # ── Recommendations ───────────────────────────────────
    recommendations = []

    if r2 is not None:
        if r2 < 0.5:
            recommendations.append("Try auto-tune --- current R² suggests suboptimal hyperparameters.")
            recommendations.append("Review feature correlation with the target (correlation heatmap).")
            recommendations.append("Consider removing low-importance features and retraining.")
        elif r2 < 0.8:
            recommendations.append("Auto-tune may improve performance further.")
            recommendations.append("Check residual plot for systematic patterns.")
        else:
            recommendations.append("Validate on a held-out or external dataset.")
            recommendations.append("Check for data leakage if R² seems unusually high.")

    if mae is not None and rmse is not None and rmse > 2 * mae:
        recommendations.append(
            "RMSE is significantly higher than MAE --- likely a few large outlier predictions. "
            "Inspect residual distribution."
        )

    # ── One-line summary for report ───────────────────────
    r2_str   = f"{r2:.4f}"   if r2   is not None else "N/A"
    mae_str  = f"{mae:.4f}"  if mae  is not None else "N/A"
    rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"

    summary_sentence = (
        f"The {model_label} achieved an R² of {r2_str} (MAE={mae_str}, RMSE={rmse_str}) "
        f"on the test set, indicating {r2_label} predictive performance."
    )

    return {
        "r2_label":        r2_label,
        "r2_badge":        r2_badge,
        "r2_text":         r2_text,
        "mae_context":     mae_context,
        "rmse_context":    rmse_context,
        "data_warnings":   data_warnings,
        "recommendations": recommendations,
        "summary_sentence": summary_sentence,
        "model_label":     model_label,
        "metrics":         metrics,
    }


def render_analysis(analysis: dict) -> None:
    """
    Render an interpret_metrics() result dict into Streamlit widgets.

    Import streamlit here (lazy) so analyze_helper stays importable
    in non-Streamlit contexts (e.g. report generation).
    """
    import streamlit as st

    st.markdown("### Model interpretation")

    # R² badge + text
    badge_html = (
        f'<span class="{analysis["r2_badge"]}">'
        f'{analysis["r2_label"].upper()}'
        f'</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown(analysis["r2_text"])

    # MAE / RMSE context
    if analysis["mae_context"]:
        st.caption(analysis["mae_context"])
    if analysis["rmse_context"]:
        st.caption(analysis["rmse_context"])

    # Data warnings
    for w in analysis["data_warnings"]:
        st.warning(w)

    # Recommendations
    if analysis["recommendations"]:
        with st.expander("Recommendations"):
            for rec in analysis["recommendations"]:
                st.markdown(f"- {rec}")