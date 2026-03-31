# pages/1_Train_Diagnostics.py
# ─────────────────────────────────────────────────────────────
# Train a single model, run diagnostics, optionally auto-tune.
# v3: ALL training settings (split, CV, autotune) live here.
# Settings are saved to session state via save_train_cfg()
# so all other pages read the same config automatically.
#
# Fixes over submitted version:
#   - hide_slow moved OUTSIDE the form so model list updates instantly
#   - Warning banner shown when settings are unsaved and user trains
#   - st.pyplot width kwarg corrected to width='stretch'
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
from backend.session_store  import save_run
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

# ══════════════════════════════════════════════════════════
# SECTION 1 — Training settings (owned by this page)
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Training settings</div>',
            unsafe_allow_html=True)
st.caption(
    "These settings apply across all pages. "
    "Click **Apply & Save** to commit them before training."
)

_existing = get_train_cfg()

# ── hide_slow OUTSIDE the form so model list reacts immediately ──
hide_slow = st.toggle(
    "Hide slow models (GP / Kernel Ridge)",
    value=_existing["hide_slow"],
    help="Both scale as O(n³) — recommended to hide for datasets > 1,000 rows.",
    key="hide_slow_toggle",
)

with st.form("training_settings_form"):
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        split_strategy = st.radio(
            "Split strategy",
            ["Time-ordered split", "Random shuffle split"],
            index=0 if _existing["split_strategy"] == "Time-ordered split" else 1,
            help=(
                "**Time-ordered**: preserves row order — use for time-series data.\n\n"
                "**Random shuffle**: random split — use for i.i.d. tabular datasets."
            ),
        )
        train_size_pct = st.slider(
            "Train size",
            min_value=50, max_value=95,
            value=int(_existing["train_size"] * 100),
            step=5, format="%d%%",
            help="Fraction of data used for training. Remainder → test set.",
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
            help="Seed for reproducible random splits.",
        )
        k_folds = st.slider(
            "CV folds (k)",
            min_value=2, max_value=10,
            value=_existing["k_folds"], step=1,
            help="Number of folds for cross-validation during auto-tune.",
        )

    with sc3:
        autotune_budget = st.select_slider(
            "Optuna trials",
            options=[5, 10, 20, 50, 100],
            value=_existing["autotune_budget"],
            help=(
                "Trials Optuna evaluates during auto-tune.\n\n"
                "5–10 → fast preview\n"
                "20–50 → good quality\n"
                "100 → thorough (slow on large models)"
            ),
        )

    submitted = st.form_submit_button("✅  Apply & Save settings", type="primary",
                                      width='stretch')

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
    st.session_state["settings_saved"] = True
    st.success(
        f"Settings saved — {int(train_size*100)}% train · "
        f"{split_strategy.split()[0]} · seed {random_seed} · "
        f"{k_folds}-fold CV · {autotune_budget} Optuna trials"
    )

# FIX: detect unsaved changes and warn before training
# We compare the form's current widget values against what is saved.
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
        "Click **✅ Apply & Save settings** before training, "
        "or training will use the previously saved configuration.",
        icon=None,
    )

# ══════════════════════════════════════════════════════════
# SECTION 2 — Model configuration
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Model configuration</div>',
            unsafe_allow_html=True)

col_model, col_info = st.columns([2, 1])

with col_model:
    # FIX: use live hide_slow toggle value (not cfg which may be stale)
    visible_keys = (
        [k for k in ALL_MODEL_KEYS if k not in ("gaussian_process", "kernel_ridge")]
        if hide_slow
        else ALL_MODEL_KEYS
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

# ── Advanced params expander ──────────────────────────────
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
        )
    st.session_state["p1_result"]   = result
    st.session_state["p1_autotune"] = None
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

# ── Plots ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Visualisations</div>',
            unsafe_allow_html=True)

tab_avp, tab_res, tab_imp = st.tabs(
    ["Actual vs Predicted", "Residuals", "Feature Importance"]
)

with tab_avp:
    fig_avp = plot_actual_vs_predicted(
        data["y_test"], data["y_pred"], target_col, labels[model_key]
    )
    st.pyplot(fig_avp, width='stretch')

with tab_res:
    fig_res = plot_residuals(data["y_test"], data["y_pred"], target_col)
    st.pyplot(fig_res, width='stretch')

with tab_imp:
    df_imp = extract_feature_importance(
        model, feat_names, data["X_train"], data["y_train"]
    )
    st.dataframe(df_imp, width='stretch', hide_index=True)
    fig_imp = plot_feature_importance(df_imp)
    st.pyplot(fig_imp, width='stretch')

# ══════════════════════════════════════════════════════════
# SECTION 5 — Auto-tune
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Auto-tune</div>', unsafe_allow_html=True)

auto_col1, auto_col2 = st.columns([3, 1])
with auto_col1:
    st.caption(
        f"Optuna will run **{cfg['autotune_budget']} trials** with "
        f"**{cfg['k_folds']}-fold CV**. "
        f"Adjust in Training Settings above and save."
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

    if not at["cv_results"].empty:
        fig_hist = plot_autotune_history(at["cv_results"])
        st.pyplot(fig_hist, width='stretch')

    fig_tuned_avp = plot_actual_vs_predicted(
        tuned["data"]["y_test"], tuned["data"]["y_pred"],
        target_col, f"{labels[model_key]} (tuned)"
    )
    st.pyplot(fig_tuned_avp, width='stretch')

# ══════════════════════════════════════════════════════════
# SECTION 6 — Save run
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Save run</div>', unsafe_allow_html=True)

save_col1, save_col2 = st.columns([3, 1])
with save_col1:
    run_notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g. baseline run, IndPenSim batch 3"
    )
with save_col2:
    save_btn = st.button("Save to comparison", width='stretch')

if save_btn:
    active_result = (
        st.session_state["p1_autotune"]["tuned_result"]
        if st.session_state.get("p1_autotune")
        else res
    )
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
        notes         = run_notes,
    )
    st.success(f"Run #{run_id} saved. View on Page 5 → Session Comparison.")