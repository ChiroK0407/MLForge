# pages/2_Model_Comparison.py
from config.sidebar_config import apply_page_config, render_sidebar, get_train_cfg
apply_page_config("Model Comparison — MLForge")

import streamlit as st
import pandas as pd
from pathlib import Path

from config.model_registry  import MODEL_REGISTRY, ALL_MODEL_KEYS
from backend.model_utils    import train_multiple_models
from backend.plotting       import plot_multi_overlay, plot_multi_scatter, plot_metrics_comparison
from backend.session_store  import save_run
from config.page_header     import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()
cfg = get_train_cfg()

st.title("Model Comparison")

if "df" not in st.session_state:
    st.info("Upload a dataset on the Home page first.", icon="⬅️")
    st.stop()

df         = st.session_state["df"]
target_col = st.session_state.get("target_col", df.columns[-1])

# ── Config summary strip ───────────────────────────────────
st.caption(
    f"Using: {cfg['split_strategy'].split()[0]} split · "
    f"{int(cfg['train_size']*100)}% train · seed {cfg['random_seed']}  "
    f"— change on **Train & Diagnostics** page."
)

# ── Model picker ───────────────────────────────────────────
st.markdown('<div class="section-header">Select models to compare</div>', unsafe_allow_html=True)

visible_keys = (
    [k for k in ALL_MODEL_KEYS if k not in ("gaussian_process", "kernel_ridge")]
    if cfg["hide_slow"]
    else ALL_MODEL_KEYS
)

groups = {}
for k in visible_keys:
    g = MODEL_REGISTRY[k]["group"]
    groups.setdefault(g, []).append(k)

selected_keys = []
gcols = st.columns(len(groups))
for i, (group, keys) in enumerate(groups.items()):
    with gcols[i]:
        st.markdown(f"**{group}**")
        for k in keys:
            if st.checkbox(MODEL_REGISTRY[k]["label"], value=(k in ["svr_rbf", "rf"]), key=f"cmp_{k}"):
                selected_keys.append(k)

if not selected_keys:
    st.warning("Select at least one model above.")
    st.stop()

# ── Run ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Run</div>', unsafe_allow_html=True)

if st.button("Run comparison", type="primary"):
    with st.spinner(f"Training {len(selected_keys)} models on shared split…"):
        results = train_multiple_models(df, target_col, selected_keys, cfg)
    st.session_state["p2_results"] = results
    st.success(f"Trained {len(results)} models.")

if "p2_results" not in st.session_state or not st.session_state["p2_results"]:
    st.stop()

results = st.session_state["p2_results"]
ok      = {k: v for k, v in results.items() if "error" not in v}
errors  = {k: v for k, v in results.items() if "error" in v}

for k, v in errors.items():
    st.error(f"{MODEL_REGISTRY[k]['label']}: {v['error']}")

if not ok:
    st.stop()

# ── Ranked metrics table ───────────────────────────────────
st.markdown('<div class="section-header">Metrics — ranked by R²</div>', unsafe_allow_html=True)

metrics_rows = []
for k, v in ok.items():
    m = v["metrics"]
    metrics_rows.append({
        "Model": MODEL_REGISTRY[k]["label"],
        "R²":    m["R2"],
        "RMSE":  m["RMSE"],
        "MAE":   m["MAE"],
        "MSE":   m["MSE"],
        "_key":  k,
    })

metrics_df = (
    pd.DataFrame(metrics_rows)
    .sort_values("R²", ascending=False)
    .reset_index(drop=True)
)
metrics_df.insert(0, "Rank", range(1, len(metrics_df) + 1))
display_df = metrics_df.drop(columns=["_key"])
st.dataframe(display_df, width='stretch', hide_index=True)

best_key   = metrics_df.iloc[0]["_key"]
best_label = MODEL_REGISTRY[best_key]["label"]
best_r2    = metrics_df.iloc[0]["R²"]
st.caption(f"Best model: **{best_label}** (R² = {best_r2:.4f})")

# ── Visual comparison ──────────────────────────────────────
st.markdown('<div class="section-header">Visual comparison</div>', unsafe_allow_html=True)

metric_choice = st.selectbox("Metric to plot", ["R²", "RMSE", "MAE", "MSE"], index=0)
metric_map    = {"R²": "R2", "RMSE": "RMSE", "MAE": "MAE", "MSE": "MSE"}
metric_key    = metric_map[metric_choice]

metrics_for_plot = {MODEL_REGISTRY[k]["label"]: v["metrics"] for k, v in ok.items()}
fig_bar = plot_metrics_comparison(metrics_for_plot, metric=metric_key)
st.pyplot(fig_bar, width='stretch')

tab_line, tab_scatter = st.tabs(["Prediction overlay", "Scatter plot"])
first_v    = next(iter(ok.values()))
y_test     = first_v["data"]["y_test"]
preds_dict = {MODEL_REGISTRY[k]["label"]: v["data"]["y_pred"] for k, v in ok.items()}

with tab_line:
    st.pyplot(plot_multi_overlay(y_test, preds_dict, target_col), width='stretch')

with tab_scatter:
    st.pyplot(plot_multi_scatter(y_test, preds_dict, target_col), width='stretch')

# ── Save best ──────────────────────────────────────────────
st.markdown('<div class="section-header">Save best model</div>', unsafe_allow_html=True)

save_col1, save_col2 = st.columns([3, 1])
with save_col1:
    save_notes = st.text_input("Notes", placeholder=f"Comparison run — best: {best_label}", key="p2_notes")
with save_col2:
    if st.button("Save best run", width='stretch'):
        best_v = ok[best_key]
        run_id = save_run(
            model_key     = best_key,
            metrics       = best_v["metrics"],
            cfg_snapshot  = cfg,
            feature_names = best_v["feature_names"],
            y_test        = best_v["data"]["y_test"],
            y_pred        = best_v["data"]["y_pred"],
            target_col    = target_col,
            notes         = save_notes or f"Best from comparison: {best_label}",
        )
        st.success(f"Best model saved as run #{run_id}.")
