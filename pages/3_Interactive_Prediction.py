# pages/3_Interactive_Prediction.py
from config.sidebar_config import apply_page_config, render_sidebar, get_train_cfg
apply_page_config("Interactive Prediction — MLForge")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config.model_registry import MODEL_REGISTRY, ALL_MODEL_KEYS
from backend.model_utils   import train_model, extract_feature_importance, retrain_reduced
from backend.plotting      import plot_feature_importance
from config.page_header    import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()
cfg = get_train_cfg()

st.title("Interactive Prediction")
st.caption("Train → select key features → predict manually with ± error bounds.")

if "df" not in st.session_state:
    st.info("Upload a dataset on the Home page first.", icon="⬅️")
    st.stop()

df         = st.session_state["df"]
target_col = st.session_state.get("target_col", df.columns[-1])

st.caption(
    f"Using: {cfg['split_strategy'].split()[0]} split · "
    f"{int(cfg['train_size']*100)}% train · seed {cfg['random_seed']}  "
    f"— change on **Train & Diagnostics** page."
)

for key in ["p3_base", "p3_importances", "p3_reduced", "p3_selected"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════
# STAGE 1 — Train baseline
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Stage 1 — baseline model</div>', unsafe_allow_html=True)

visible_keys = (
    [k for k in ALL_MODEL_KEYS if k not in ("gaussian_process", "kernel_ridge")]
    if cfg["hide_slow"]
    else ALL_MODEL_KEYS
)

model_key = st.selectbox(
    "Model",
    options=visible_keys,
    format_func=lambda k: MODEL_REGISTRY[k]["label"],
    key="p3_model",
)

if st.button("Train baseline", type="primary"):
    with st.spinner("Training…"):
        result = train_model(df, target_col, model_key, cfg)
        df_imp = extract_feature_importance(
            result["model"],
            result["feature_names"],
            result["data"]["X_train"],
            result["data"]["y_train"],
        )
    st.session_state["p3_base"]        = result
    st.session_state["p3_importances"] = df_imp
    st.session_state["p3_reduced"]     = None
    st.session_state["p3_selected"]    = None
    st.success(f"Baseline trained — R²={result['metrics']['R2']:.4f}, RMSE={result['metrics']['RMSE']:.4f}")

if st.session_state["p3_base"]:
    base = st.session_state["p3_base"]
    m1, m2, m3 = st.columns(3)
    m1.metric("R²",   f"{base['metrics']['R2']:.4f}")
    m2.metric("RMSE", f"{base['metrics']['RMSE']:.4f}")
    m3.metric("MAE",  f"{base['metrics']['MAE']:.4f}")

    st.markdown("**Feature importances (normalised)**")
    df_imp = st.session_state["p3_importances"]
    col_tbl, col_plt = st.columns([1, 1])
    with col_tbl:
        st.dataframe(df_imp, width='stretch', hide_index=True)
    with col_plt:
        fig_imp = plot_feature_importance(df_imp, top_n=12)
        st.pyplot(fig_imp, width='stretch')

# ══════════════════════════════════════════════════════════
# STAGE 2 — Feature selection + retrain
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Stage 2 — feature selection</div>', unsafe_allow_html=True)

if st.session_state["p3_importances"] is None:
    st.info("Complete Stage 1 first.")
else:
    df_imp       = st.session_state["p3_importances"]
    all_features = df_imp["feature"].tolist()
    default_top  = all_features[:min(3, len(all_features))]

    selected = st.multiselect(
        "Select features for the reduced model",
        options=all_features,
        default=st.session_state["p3_selected"] or default_top,
        help="Tip: start with the top 3–5 by importance.",
    )
    st.session_state["p3_selected"] = selected

    if selected and st.button("Retrain with selected features"):
        with st.spinner("Retraining reduced model…"):
            reduced = retrain_reduced(df, target_col, model_key, cfg, selected)
        st.session_state["p3_reduced"] = reduced
        st.success(f"Reduced model — R²={reduced['metrics']['R2']:.4f}, RMSE={reduced['metrics']['RMSE']:.4f}")

    if st.session_state["p3_reduced"]:
        red = st.session_state["p3_reduced"]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("R² (reduced)",   f"{red['metrics']['R2']:.4f}")
        rc2.metric("RMSE (reduced)",  f"{red['metrics']['RMSE']:.4f}")
        rc3.metric("MAE (reduced)",   f"{red['metrics']['MAE']:.4f}")

# ══════════════════════════════════════════════════════════
# STAGE 3 — Manual input + prediction
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Stage 3 — predict</div>', unsafe_allow_html=True)

if st.session_state["p3_reduced"] is None:
    st.info("Complete Stage 2 (retrain) first.")
else:
    red      = st.session_state["p3_reduced"]
    selected = st.session_state["p3_selected"] or []
    rmse     = red["rmse"]

    st.caption("Enter values for each feature. Ranges based on the uploaded dataset.")

    input_vals = {}
    n_cols     = min(3, len(selected))
    rows       = [selected[i:i+n_cols] for i in range(0, len(selected), n_cols)]

    for row_feats in rows:
        inp_cols = st.columns(len(row_feats))
        for col, feat in zip(inp_cols, row_feats):
            with col:
                fmin  = float(df[feat].min())
                fmax  = float(df[feat].max())
                fmean = float(df[feat].mean())
                step  = max((fmax - fmin) / 100, 1e-6)
                input_vals[feat] = st.number_input(
                    f"{feat}",
                    min_value=fmin, max_value=fmax, value=fmean,
                    step=step, format="%.4f",
                    help=f"Range: [{fmin:.4f}, {fmax:.4f}]",
                )

    if st.button("Predict", type="primary"):
        X_manual = pd.DataFrame([input_vals])[selected]
        X_scaled = red["scaler"].transform(X_manual) if red.get("scaler") else X_manual.values
        y_hat    = float(red["model"].predict(X_scaled)[0])

        st.markdown("---")
        pred_col, bound_col = st.columns(2)
        with pred_col:
            st.metric(f"Predicted {target_col}", f"{y_hat:.4f}")
        with bound_col:
            st.metric("± RMSE bound", f"[{y_hat - rmse:.4f},  {y_hat + rmse:.4f}]")

        if y_hat < float(df[target_col].min()) or y_hat > float(df[target_col].max()):
            st.warning("Prediction is outside the training data range — extrapolation may be unreliable.")
