# pages/5_Session_Comparison.py
from config.sidebar_config import apply_page_config, render_sidebar
apply_page_config("Session Comparison — MLForge")

import streamlit as st
import pandas as pd
from pathlib import Path

from config.model_registry import MODEL_REGISTRY
from backend.session_store import get_runs, runs_to_dataframe, clear_runs, delete_run, get_best_run
from backend.plotting      import plot_multi_overlay, plot_multi_scatter, plot_metrics_comparison
from config.page_header    import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()
runs = get_runs()

st.title("Session Comparison")

if not runs:
    st.info("No runs saved yet. Train a model on Page 1 or 2 and click **Save run**.", icon="💾")
    st.stop()

# ── Summary strip ──────────────────────────────────────────
best = get_best_run("R2")
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Total runs saved", len(runs))
sc2.metric("Best R²",   f"{best['metrics']['R2']:.4f}" if best else "—")
sc3.metric("Best model", best["model_label"] if best else "—")

# ── Comparison table ───────────────────────────────────────
st.markdown('<div class="section-header">All runs</div>', unsafe_allow_html=True)

comp_df = runs_to_dataframe(runs)

if "ΔR² vs #1" in comp_df.columns:
    def colour_delta(val):
        if pd.isna(val) or val == 0:
            return ""
        return "color: #1a9e5c" if val > 0 else "color: #c0392b"
    styled = comp_df.style.applymap(colour_delta, subset=["ΔR² vs #1"])
    st.dataframe(styled, width='stretch', hide_index=True)
else:
    st.dataframe(comp_df, width='stretch', hide_index=True)

# ── Visual comparison ──────────────────────────────────────
st.markdown('<div class="section-header">Visual comparison</div>', unsafe_allow_html=True)

plottable = [r for r in runs if r.get("y_test") is not None]

if len(plottable) >= 2:
    datasets = {r["dataset_name"] for r in plottable}
    if len(datasets) > 1:
        st.warning(
            f"Runs span multiple datasets ({', '.join(datasets)}). "
            "Overlay plots may be misleading."
        )

    metric_choice = st.selectbox("Metric bar chart", ["R2", "RMSE", "MAE", "MSE"], index=0, key="p5_metric")
    metrics_for_plot = {f"#{r['run_id']} {r['model_label']}": r["metrics"] for r in plottable}
    fig_bar = plot_metrics_comparison(metrics_for_plot, metric=metric_choice)
    st.pyplot(fig_bar, width='stretch')

    first_len = len(plottable[0]["y_test"])
    same_len  = [r for r in plottable if len(r["y_test"]) == first_len]

    if len(same_len) >= 2:
        y_test     = same_len[0]["y_test"]
        preds_dict = {f"#{r['run_id']} {r['model_label']}": r["y_pred"] for r in same_len}
        target_col = same_len[0].get("target_col", "target")

        tab_l, tab_s = st.tabs(["Prediction overlay", "Scatter"])
        with tab_l:
            st.pyplot(plot_multi_overlay(y_test, preds_dict, target_col), width='stretch')
        with tab_s:
            st.pyplot(plot_multi_scatter(y_test, preds_dict, target_col), width='stretch')
    else:
        st.caption("Overlay plots require runs with the same test set size.")
else:
    st.caption("Save at least 2 runs to enable visual comparison.")

# ── Per-run detail cards ───────────────────────────────────
st.markdown('<div class="section-header">Run details</div>', unsafe_allow_html=True)

for run in runs:
    with st.expander(
        f"Run #{run['run_id']} · {run['model_label']} · "
        f"R²={run['metrics'].get('R2', 0):.4f} · {run['timestamp']}"
    ):
        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("**Metrics**")
            st.dataframe(pd.DataFrame([run["metrics"]]), width='stretch', hide_index=True)
            if run.get("best_params"):
                st.markdown("**Best params (auto-tuned)**")
                st.dataframe(pd.DataFrame([run["best_params"]]), width='stretch', hide_index=True)

        with dc2:
            st.markdown("**Config**")
            cfg_snap = run.get("cfg_snapshot", {})
            st.json({
                "train_size":     cfg_snap.get("train_size"),
                "split_strategy": cfg_snap.get("split_strategy"),
                "k_folds":        cfg_snap.get("k_folds"),
                "random_seed":    cfg_snap.get("random_seed"),
            }, expanded=False)
            st.markdown(f"**Dataset:** {run.get('dataset_name', '—')}")
            st.markdown(f"**Target:** {run.get('target_col', '—')}")
            if run.get("notes"):
                st.markdown(f"**Notes:** {run['notes']}")

        if st.button(f"Delete run #{run['run_id']}", key=f"del_{run['run_id']}"):
            delete_run(run["run_id"])
            st.rerun()

st.divider()
if st.button("Clear all runs", type="secondary"):
    clear_runs()
    st.rerun()
