# pages/4_General_Testing.py
from config.sidebar_config import apply_page_config, render_sidebar, get_train_cfg
apply_page_config("General Testing — MLForge")

import streamlit as st
import pandas as pd
from pathlib import Path

from config.model_registry  import MODEL_REGISTRY, ALL_MODEL_KEYS
from backend.model_utils    import train_model, train_multiple_models
from backend.plotting       import (
    plot_actual_vs_predicted, plot_residuals,
    plot_multi_overlay, plot_multi_scatter, plot_metrics_comparison,
)
from backend.analyze_helper import interpret_metrics, render_analysis
from backend.session_store  import save_run
from config.page_header     import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()
cfg = get_train_cfg()

st.title("General Testing")
st.caption("Test your models against any dataset — no Home page upload required.")

# ── Dataset upload ─────────────────────────────────────────
st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Can be any tabular regression dataset.",
)

if not uploaded:
    if "df" in st.session_state:
        st.info(f"No file uploaded — using active dataset: **{st.session_state.get('active_dataset', 'unknown')}**")
        df = st.session_state["df"]
    else:
        st.info("Upload a dataset above or load one on the Home page.", icon="⬆️")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.success(f"{uploaded.name} — {len(df):,} rows × {len(df.columns)} cols")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

col_tgt, col_prev = st.columns([1, 2])
with col_tgt:
    target_col = st.selectbox("Target column", numeric_cols)
with col_prev:
    st.dataframe(df.head(5), width='stretch')

st.caption(
    f"Using: {cfg['split_strategy'].split()[0]} split · "
    f"{int(cfg['train_size']*100)}% train · seed {cfg['random_seed']}  "
    f"— change on **Train & Diagnostics** page."
)

# ── Testing mode ───────────────────────────────────────────
st.markdown('<div class="section-header">Testing mode</div>', unsafe_allow_html=True)

mode = st.radio("Mode", ["Single model", "Compare all selected models"], horizontal=True)

visible_keys = (
    [k for k in ALL_MODEL_KEYS if k not in ("gaussian_process", "kernel_ridge")]
    if cfg["hide_slow"]
    else ALL_MODEL_KEYS
)

if mode == "Single model":
    model_key = st.selectbox("Model", options=visible_keys, format_func=lambda k: MODEL_REGISTRY[k]["label"])

    if st.button("Run test", type="primary"):
        with st.spinner(f"Training {MODEL_REGISTRY[model_key]['label']}…"):
            result = train_model(df, target_col, model_key, cfg)
        st.session_state["p4_single"] = result
        st.session_state["p4_mode"]   = "single"

    if st.session_state.get("p4_mode") == "single" and st.session_state.get("p4_single"):
        res     = st.session_state["p4_single"]
        metrics = res["metrics"]
        data    = res["data"]

        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("R²",   f"{metrics['R2']:.4f}")
        mc2.metric("RMSE", f"{metrics['RMSE']:.4f}")
        mc3.metric("MAE",  f"{metrics['MAE']:.4f}")
        mc4.metric("MSE",  f"{metrics['MSE']:.6f}")

        analysis = interpret_metrics(metrics, model_key, df, target_col)
        render_analysis(analysis)

        tab1, tab2 = st.tabs(["Actual vs Predicted", "Residuals"])
        with tab1:
            st.pyplot(
                plot_actual_vs_predicted(data["y_test"], data["y_pred"], target_col, MODEL_REGISTRY[model_key]["label"]),
                width='stretch',
            )
        with tab2:
            st.pyplot(plot_residuals(data["y_test"], data["y_pred"], target_col), width='stretch')

        with st.expander("Sample predictions (first 30 rows)"):
            preview = pd.DataFrame({
                "Actual":    data["y_test"][:30],
                "Predicted": data["y_pred"][:30],
                "Residual":  data["y_test"][:30] - data["y_pred"][:30],
            })
            st.dataframe(preview, width='stretch', hide_index=True)

        if st.button("Save this run"):
            rid = save_run(
                model_key     = model_key,
                metrics       = metrics,
                cfg_snapshot  = cfg,
                feature_names = res["feature_names"],
                y_test        = data["y_test"],
                y_pred        = data["y_pred"],
                target_col    = target_col,
                dataset_name  = getattr(uploaded, "name", "general"),
                notes         = "General testing — single model",
            )
            st.success(f"Saved as run #{rid}.")

else:
    selected = st.multiselect(
        "Select models",
        options=visible_keys,
        default=["svr_rbf", "rf", "ridge"],
        format_func=lambda k: MODEL_REGISTRY[k]["label"],
    )
    if not selected:
        st.warning("Pick at least one model.")
        st.stop()

    if st.button("Run comparison", type="primary"):
        with st.spinner(f"Training {len(selected)} models…"):
            results = train_multiple_models(df, target_col, selected, cfg)
        st.session_state["p4_multi"] = results
        st.session_state["p4_mode"]  = "multi"

    if st.session_state.get("p4_mode") == "multi" and st.session_state.get("p4_multi"):
        results = st.session_state["p4_multi"]
        ok      = {k: v for k, v in results.items() if "error" not in v}
        errors  = {k: v for k, v in results.items() if "error" in v}

        for k, v in errors.items():
            st.error(f"{MODEL_REGISTRY[k]['label']}: {v['error']}")

        if not ok:
            st.stop()

        st.markdown('<div class="section-header">Ranked results</div>', unsafe_allow_html=True)
        rows = [{
            "Rank":  i + 1,
            "Model": MODEL_REGISTRY[k]["label"],
            "R²":    v["metrics"]["R2"],
            "RMSE":  v["metrics"]["RMSE"],
            "MAE":   v["metrics"]["MAE"],
            "MSE":   v["metrics"]["MSE"],
        } for i, (k, v) in enumerate(
            sorted(ok.items(), key=lambda x: x[1]["metrics"]["R2"], reverse=True)
        )]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        metrics_for_plot = {MODEL_REGISTRY[k]["label"]: v["metrics"] for k, v in ok.items()}
        st.pyplot(plot_metrics_comparison(metrics_for_plot, metric="R2"), width='stretch')

        first_v    = next(iter(ok.values()))
        y_test     = first_v["data"]["y_test"]
        preds_dict = {MODEL_REGISTRY[k]["label"]: v["data"]["y_pred"] for k, v in ok.items()}

        tab_l, tab_s = st.tabs(["Overlay", "Scatter"])
        with tab_l:
            st.pyplot(plot_multi_overlay(y_test, preds_dict, target_col), width='stretch')
        with tab_s:
            st.pyplot(plot_multi_scatter(y_test, preds_dict, target_col), width='stretch')

        with st.expander("Model interpretations"):
            for k, v in ok.items():
                st.markdown(f"**{MODEL_REGISTRY[k]['label']}**")
                analysis = interpret_metrics(v["metrics"], k, df, target_col)
                st.markdown(analysis["summary_sentence"])
                st.divider()
