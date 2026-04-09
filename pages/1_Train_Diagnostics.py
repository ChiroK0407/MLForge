# pages/1_Train_Diagnostics.py
# ─────────────────────────────────────────────────────────────
# Train a single model, run diagnostics, optionally auto-tune.
# v4: Smart Save — user selects which artifacts to store.
#     Stored artifacts flow directly into the report generator.
# ─────────────────────────────────────────────────────────────

from config.sidebar_config import (
    apply_page_config, render_sidebar, save_train_cfg, get_train_cfg
)
apply_page_config("Train & Diagnostics — MLForge")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config.model_registry  import MODEL_REGISTRY, ALL_MODEL_KEYS
from backend.model_utils    import train_model, extract_feature_importance
from backend.autotune       import run_autotune
from backend.plotting       import (
    plot_actual_vs_predicted, plot_residuals,
    plot_feature_importance, plot_autotune_history,
)
from backend.analyze_helper import interpret_metrics, render_analysis
from backend.session_store  import save_run, figure_to_bytes, ARTIFACT_LABELS
from config.page_header     import render_header

css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()

st.title("Train & Diagnostics")

if "df" not in st.session_state:
    st.info("Upload a dataset on the Home page first.", icon="⬅️")
    st.stop()

df         = st.session_state["df"]
target_col = st.session_state.get("target_col", df.columns[-1])
input_features = st.session_state.get("input_features")

# ══════════════════════════════════════════════════════════
# SECTION 1 — Training settings
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Training settings</div>',
            unsafe_allow_html=True)
st.caption(
    "These settings apply across all pages. "
    "Click **Apply & Save** to commit them before training."
)

_existing = get_train_cfg()

# hide_slow outside form so model list reacts immediately
hide_slow = st.toggle(
    "Hide slow models (GP / Kernel Ridge)",
    value=_existing["hide_slow"],
    help="Both scale as O(n³) — recommended for datasets > 1,000 rows.",
    key="hide_slow_toggle",
)

with st.form("training_settings_form"):
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        split_strategy = st.radio(
            "Split strategy",
            ["Time-ordered split", "Random shuffle split"],
            index=0 if _existing["split_strategy"] == "Time-ordered split" else 1,
        )
        train_size_pct = st.slider(
            "Train size",
            min_value=50, max_value=95,
            value=int(_existing["train_size"] * 100),
            step=5, format="%d%%",
        )
        train_size = train_size_pct / 100.0
        if st.session_state.get("dataset_row_count"):
            n       = st.session_state["dataset_row_count"]
            n_train = int(train_size * n)
            n_test  = n - n_train
            st.caption(f"→ {n_train:,} train · {n_test:,} test samples")

    with sc2:
        random_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=99999,
            value=_existing["random_seed"], step=1,
        )
        k_folds = st.slider(
            "CV folds (k)",
            min_value=2, max_value=10,
            value=_existing["k_folds"], step=1,
        )

    with sc3:
        autotune_budget = st.select_slider(
            "Optuna trials",
            options=[5, 10, 20, 50, 100],
            value=_existing["autotune_budget"],
        )

    submitted = st.form_submit_button(
        "✅  Apply & Save settings", type="primary", width='stretch'
    )

if submitted:
    new_cfg = {
        "train_size":      train_size,
        "test_size":       round(1.0 - train_size, 2),
        "k_folds":         int(k_folds),
        "split_strategy":  split_strategy,
        "random_seed":     int(random_seed),
        "autotune_budget": int(autotune_budget),
        "hide_slow":       hide_slow,
    }
    save_train_cfg(new_cfg)
    st.success(
        f"Settings saved — {int(train_size*100)}% train · "
        f"{split_strategy.split()[0]} · seed {random_seed} · "
        f"{k_folds}-fold CV · {autotune_budget} Optuna trials"
    )

cfg = get_train_cfg()

_form_matches_cfg = (
    split_strategy == cfg["split_strategy"]
    and abs(train_size - cfg["train_size"]) < 0.001
    and int(random_seed) == cfg["random_seed"]
    and int(k_folds) == cfg["k_folds"]
    and int(autotune_budget) == cfg["autotune_budget"]
    and hide_slow == cfg["hide_slow"]
)
if not _form_matches_cfg:
    st.warning(
        "⚠️ You have unsaved setting changes. "
        "Click **✅ Apply & Save settings** before training.",
        icon=None,
    )

# ══════════════════════════════════════════════════════════
# SECTION 2 — Model configuration
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Model configuration</div>',
            unsafe_allow_html=True)

col_model, col_info = st.columns([2, 1])

with col_model:
    visible_keys = (
        [k for k in ALL_MODEL_KEYS if k not in ("gaussian_process", "kernel_ridge")]
        if hide_slow else ALL_MODEL_KEYS
    )
    labels    = {k: MODEL_REGISTRY[k]["label"] for k in visible_keys}
    model_key = st.selectbox(
        "Select model",
        options=visible_keys,
        format_func=lambda k: labels[k],
    )

with col_info:
    meta = MODEL_REGISTRY[model_key]
    st.markdown(f"**Group:** {meta['group']}")
    st.caption(meta["description"])
    st.caption(
        f"Split: {cfg['split_strategy'].split()[0]}  "
        f"· Train: {int(cfg['train_size']*100)}%  "
        f"· Seed: {cfg['random_seed']}"
    )

with st.expander("Override model hyperparameters (optional)"):
    default_params = meta["default_params"].copy()
    custom_params  = {}
    param_cols     = st.columns(min(len(default_params), 3))
    for i, (param, default_val) in enumerate(default_params.items()):
        with param_cols[i % len(param_cols)]:
            if isinstance(default_val, float):
                custom_params[param] = st.number_input(
                    param, value=float(default_val),
                    format="%.5f",
                    step=float(default_val) * 0.1 or 0.001,
                    key=f"p1_{param}",
                )
            elif isinstance(default_val, int):
                custom_params[param] = st.number_input(
                    param, value=int(default_val), step=1, key=f"p1_{param}"
                )
            elif isinstance(default_val, str):
                custom_params[param] = st.text_input(
                    param, value=default_val, key=f"p1_{param}"
                )
            else:
                custom_params[param] = default_val
    use_custom = st.toggle("Apply custom params", value=False)

# ══════════════════════════════════════════════════════════
# SECTION 3 — Train
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Training</div>', unsafe_allow_html=True)

if st.button("Train model", type="primary"):
    with st.spinner(f"Training {labels[model_key]}…"):
        result = train_model(
            df, target_col, model_key, cfg,
            custom_params=custom_params if use_custom else None,
            input_features=input_features,
        )
    st.session_state["p1_result"]   = result
    st.session_state["p1_autotune"] = None
    # Clear previously cached figures so they are regenerated fresh
    for k in ["p1_fig_avp", "p1_fig_res", "p1_fig_imp", "p1_df_imp"]:
        st.session_state.pop(k, None)
    st.success("Training complete.")

if "p1_result" not in st.session_state or st.session_state["p1_result"] is None:
    st.stop()

res        = st.session_state["p1_result"]
data       = res["data"]
metrics    = res["metrics"]
model      = res["model"]
feat_names = res["feature_names"]

# ══════════════════════════════════════════════════════════
# SECTION 4 — Results
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Test set metrics</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²",   f"{metrics['R2']:.4f}")
m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
m3.metric("MAE",  f"{metrics['MAE']:.4f}")
m4.metric("MSE",  f"{metrics['MSE']:.6f}")

analysis = interpret_metrics(metrics, model_key, df, target_col)
render_analysis(analysis)

# ── Plots (cached in session state after first render) ────
st.markdown('<div class="section-header">Visualisations</div>',
            unsafe_allow_html=True)

tab_avp, tab_res, tab_imp = st.tabs(
    ["Actual vs Predicted", "Residuals", "Feature Importance"]
)

with tab_avp:
    if "p1_fig_avp" not in st.session_state:
        st.session_state["p1_fig_avp"] = plot_actual_vs_predicted(
            data["y_test"], data["y_pred"], target_col, labels[model_key]
        )
    st.pyplot(st.session_state["p1_fig_avp"], width='stretch')

with tab_res:
    if "p1_fig_res" not in st.session_state:
        st.session_state["p1_fig_res"] = plot_residuals(
            data["y_test"], data["y_pred"], target_col
        )
    st.pyplot(st.session_state["p1_fig_res"], width='stretch')

with tab_imp:
    if "p1_df_imp" not in st.session_state:
        st.session_state["p1_df_imp"] = extract_feature_importance(
            model, feat_names, data["X_train"], data["y_train"]
        )
    if "p1_fig_imp" not in st.session_state:
        st.session_state["p1_fig_imp"] = plot_feature_importance(
            st.session_state["p1_df_imp"]
        )
    st.dataframe(
        st.session_state["p1_df_imp"], width='stretch', hide_index=True
    )
    st.pyplot(st.session_state["p1_fig_imp"], width='stretch')

# ══════════════════════════════════════════════════════════
# SECTION 5 — Auto-tune
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Auto-tune</div>', unsafe_allow_html=True)

auto_col1, auto_col2 = st.columns([3, 1])
with auto_col1:
    st.caption(
        f"Optuna will run **{cfg['autotune_budget']} trials** with "
        f"**{cfg['k_folds']}-fold CV**. Adjust in Training Settings above."
    )
with auto_col2:
    run_autotune_btn = st.button("Run auto-tune", width='stretch')

if run_autotune_btn:
    progress_bar = st.progress(0, text="Starting…")

    def _cb(done, total):
        progress_bar.progress(done / total, text=f"Trial {done}/{total}")

    with st.spinner("Searching hyperparameters…"):
        autotune_result = run_autotune(
            model_key,
            data["X_train"], data["y_train"],
            cfg,
            budget=cfg["autotune_budget"],
            progress_callback=_cb,
        )
    progress_bar.empty()

    if autotune_result.get("error"):
        st.error(autotune_result["error"])
    else:
        tuned_result = train_model(
            df, target_col, model_key, cfg,
            custom_params=autotune_result["best_params"],
        )
        autotune_result["tuned_result"] = tuned_result
        # Cache autotune figures
        if not autotune_result["cv_results"].empty:
            st.session_state["p1_fig_at_hist"] = plot_autotune_history(
                autotune_result["cv_results"]
            )
        st.session_state["p1_fig_at_avp"] = plot_actual_vs_predicted(
            tuned_result["data"]["y_test"], tuned_result["data"]["y_pred"],
            target_col, f"{labels[model_key]} (tuned)"
        )
        st.session_state["p1_autotune"] = autotune_result
        st.success(
            f"Best CV R² = **{autotune_result['best_score']:.4f}** "
            f"via {autotune_result['method']} "
            f"({autotune_result['n_trials']} trials)"
        )

if st.session_state.get("p1_autotune"):
    at     = st.session_state["p1_autotune"]
    tuned  = at["tuned_result"]
    t_mets = tuned["metrics"]

    st.markdown("#### Hyperparameters found")
    st.dataframe(
        pd.DataFrame([at["best_params"]]),
        width='stretch', hide_index=True
    )

    st.markdown("#### Metrics: original vs tuned")
    compare_df = pd.DataFrame({
        "Metric":   ["R²",          "RMSE",          "MAE",          "MSE"],
        "Original": [metrics["R2"], metrics["RMSE"], metrics["MAE"], metrics["MSE"]],
        "Tuned":    [t_mets["R2"],  t_mets["RMSE"],  t_mets["MAE"],  t_mets["MSE"]],
    })
    compare_df["Δ"] = (compare_df["Tuned"] - compare_df["Original"]).round(6)
    st.dataframe(compare_df, width='stretch', hide_index=True)

    if "p1_fig_at_hist" in st.session_state:
        st.pyplot(st.session_state["p1_fig_at_hist"], width='stretch')

    if "p1_fig_at_avp" in st.session_state:
        st.pyplot(st.session_state["p1_fig_at_avp"], width='stretch')

# ══════════════════════════════════════════════════════════
# SECTION 6 — Smart Save
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Save run</div>', unsafe_allow_html=True)
st.markdown(
    "Choose which results to store with this run. "
    "Only saved artifacts will be available in the **Report Generator**."
)

# ── Build available options dynamically ───────────────────
_available: dict[str, str] = {
    "actual_vs_predicted": ARTIFACT_LABELS["actual_vs_predicted"],
    "residuals":           ARTIFACT_LABELS["residuals"],
}
if "p1_fig_imp" in st.session_state:
    _available["feature_importance"] = ARTIFACT_LABELS["feature_importance"]

if st.session_state.get("p1_autotune"):
    _available["autotune_metrics"]    = ARTIFACT_LABELS["autotune_metrics"]
    _available["autotune_comparison"] = ARTIFACT_LABELS["autotune_comparison"]
    if "p1_fig_at_hist" in st.session_state:
        _available["autotune_history"] = ARTIFACT_LABELS["autotune_history"]

_default_keys = list(_available.keys())

sv1, sv2 = st.columns([3, 1])

with sv1:
    selected_artifacts = st.multiselect(
        "Artifacts to save",
        options=list(_available.keys()),
        default=_default_keys,
        format_func=lambda k: _available[k],
        help=(
            "All checked items will be stored with the run and made available "
            "as options in the Report Generator."
        ),
    )
    run_notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g. baseline run, IndPenSim batch 3"
    )

with sv2:
    st.markdown("&nbsp;")   # vertical alignment spacer
    save_btn = st.button("Save run", type="primary", width='stretch')

if save_btn:
    # Determine which result to save (tuned if available, else original)
    active_result = (
        st.session_state["p1_autotune"]["tuned_result"]
        if st.session_state.get("p1_autotune")
        else res
    )

    # Build artifacts dict from selected keys
    artifacts: dict = {}

    if "actual_vs_predicted" in selected_artifacts:
        fig = st.session_state.get("p1_fig_avp")
        if fig:
            artifacts["actual_vs_predicted"] = figure_to_bytes(fig)

    if "residuals" in selected_artifacts:
        fig = st.session_state.get("p1_fig_res")
        if fig:
            artifacts["residuals"] = figure_to_bytes(fig)

    if "feature_importance" in selected_artifacts:
        fig = st.session_state.get("p1_fig_imp")
        if fig:
            artifacts["feature_importance"] = figure_to_bytes(fig)

    if "autotune_history" in selected_artifacts:
        fig = st.session_state.get("p1_fig_at_hist")
        if fig:
            artifacts["autotune_history"] = figure_to_bytes(fig)

    if "autotune_metrics" in selected_artifacts and st.session_state.get("p1_autotune"):
        at = st.session_state["p1_autotune"]
        artifacts["autotune_metrics"] = {
            "best_params": at.get("best_params", {}),
            "best_score":  at.get("best_score"),
            "method":      at.get("method", ""),
            "n_trials":    at.get("n_trials", 0),
        }

    if "autotune_comparison" in selected_artifacts and st.session_state.get("p1_autotune"):
        at     = st.session_state["p1_autotune"]
        tuned  = at["tuned_result"]
        t_mets = tuned["metrics"]
        artifacts["autotune_comparison"] = [
            {"Metric": "R²",   "Original": metrics["R2"],   "Tuned": t_mets["R2"],   "Δ": round(t_mets["R2"]   - metrics["R2"],   6)},
            {"Metric": "RMSE", "Original": metrics["RMSE"], "Tuned": t_mets["RMSE"], "Δ": round(t_mets["RMSE"] - metrics["RMSE"], 6)},
            {"Metric": "MAE",  "Original": metrics["MAE"],  "Tuned": t_mets["MAE"],  "Δ": round(t_mets["MAE"]  - metrics["MAE"],  6)},
            {"Metric": "MSE",  "Original": metrics["MSE"],  "Tuned": t_mets["MSE"],  "Δ": round(t_mets["MSE"]  - metrics["MSE"],  6)},
        ]

    run_id = save_run(
        model_key     = model_key,
        metrics       = active_result["metrics"],
        cfg_snapshot  = cfg,
        feature_names = active_result["feature_names"],
        y_test        = active_result["data"]["y_test"],
        y_pred        = active_result["data"]["y_pred"],
        best_params   = st.session_state["p1_autotune"]["best_params"]
                        if st.session_state.get("p1_autotune") else None,
        target_col    = target_col,
        input_features= input_features,
        notes         = run_notes,
        artifacts     = artifacts,
    )

    saved_labels = [_available[k] for k in selected_artifacts if k in _available]
    st.success(
        f"Run #{run_id} saved with {len(artifacts)} artifact(s): "
        f"{', '.join(saved_labels) or 'metrics only'}. "
        f"View on Page 5 → Session Comparison."
    )

# ── Clear/Refresh ──────────────────────────────────────────
st.markdown("---")
col_clear, col_spacer = st.columns([1, 3])
with col_clear:
    if st.button("🔄 Clear & Refresh", help="Keep dataset loaded, clear training results, and return to model selection"):
        # Clear training-related session state
        keys_to_clear = [
            "p1_result", "p1_autotune", "p1_fig_avp", "p1_fig_res", 
            "p1_fig_imp", "p1_df_imp", "p1_fig_at_hist", "p1_fig_at_avp"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()
