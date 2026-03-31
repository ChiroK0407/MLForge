# Home.py
# ─────────────────────────────────────────────────────────────
# MLForge — main entry point.
# Layout: header → what this app does → model index → upload
# Dataset stored globally in session state once uploaded.
# ─────────────────────────────────────────────────────────────

from config.sidebar_config import apply_page_config, render_sidebar
apply_page_config("MLForge — Home")

import streamlit as st
import pandas as pd
from pathlib import Path
from backend.dataset_profiler import profile_dataset
from config.page_header import render_header

# ── Global CSS ─────────────────────────────────────────────
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

render_sidebar()
render_header()

# ══════════════════════════════════════════════════════════
# SECTION 1 — What is MLForge?
# ══════════════════════════════════════════════════════════
st.markdown("## What is MLForge?")
st.markdown(
    """
    MLForge is a general-purpose **supervised regression training platform**
    that lets you upload any tabular CSV or Excel dataset, train across
    **21 algorithms in 7 model families**, compare results, and export a
    formatted report — with no code changes required.

    Originally built around hand-coded SVR implementations for the **IndPenSim**
    penicillin fermentation dataset, it has been extended to cover the full range
    of ML algorithms reported across hydrogen production and biofuel literature.
    """
)

st.divider()

# ══════════════════════════════════════════════════════════
# SECTION 2 — How to use (page guide)
# ══════════════════════════════════════════════════════════
st.markdown("## How to use MLForge")
st.markdown(
    "Navigate using the sidebar. Upload your dataset here first — "
    "every other page shares it automatically."
)

pages_info = [
    ("🏠", "Home",
     "Upload your dataset once — all pages share it."),
    ("⚗️", "Train & Diagnostics",
     "Train a single model. Set split ratio, CV folds, and autotune budget here."),
    ("📊", "Model Comparison",
     "Train multiple models on the same split. Fair, reproducible comparison."),
    ("🎯", "Interactive Prediction",
     "Select key features, retrain a reduced model, predict manually with ± error bounds."),
    ("🧪", "General Testing",
     "Test your models against any external dataset — no re-upload of the main data."),
    ("🗂️", "Session Comparison",
     "View all saved runs side by side with delta R² vs baseline."),
    ("📄", "Report Generator",
     "Export results as Word (.docx), PDF, or LaTeX (.tex) — ready for submission."),
]

for icon, name, desc in pages_info:
    pc1, pc2 = st.columns([1, 11])
    with pc1:
        st.markdown(
            f"<div style='font-size:1.4rem; padding-top:0.1rem;'>{icon}</div>",
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(f"**{name}** — {desc}")

st.divider()

# ══════════════════════════════════════════════════════════
# SECTION 3 — Available models index
# ══════════════════════════════════════════════════════════
st.markdown("## Available models")
from config.model_registry import MODEL_REGISTRY, get_model_names_by_group

n_models = len(MODEL_REGISTRY)
st.markdown(
    f"**{n_models} models** across 7 families — covering the most frequently "
    "reported algorithms in regression and ML literature."
)

group_display_order = [
    "SVR", "Tree-Based", "Ensemble", "Boosting",
    "Linear", "Neural Network", "Probabilistic", "Other",
]
groups_by_key = get_model_names_by_group()

for group_label in group_display_order:
    keys = groups_by_key.get(group_label, [])
    if not keys:
        continue
    st.markdown(f"#### {group_label}")
    st.dataframe(
        pd.DataFrame([
            {
                "Key":         k,
                "Model":       MODEL_REGISTRY[k]["label"],
                "Description": MODEL_REGISTRY[k]["description"],
            }
            for k in keys
        ]),
        width='stretch',
        hide_index=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════
# SECTION 4 — Upload (action, intentionally last)
# ══════════════════════════════════════════════════════════
st.markdown("## Load your dataset")
st.markdown(
    "Upload once here — every page in the app will share this dataset. "
    "Re-upload at any time to switch to a different file."
)

if st.session_state.get("active_dataset"):
    st.success(
        f"✓  **{st.session_state['active_dataset']}** is currently loaded — "
        f"{st.session_state.get('dataset_row_count', '?'):,} rows × "
        f"{st.session_state.get('dataset_col_count', '?')} columns. "
        f"Re-upload below to switch datasets."
    )

uploaded = st.file_uploader(
    "Drag and drop a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Any tabular regression dataset. 200 MB limit.",
)

if uploaded:
    try:
        df = (
            pd.read_csv(uploaded)
            if uploaded.name.endswith(".csv")
            else pd.read_excel(uploaded)
        )
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.session_state["df"]                = df
    st.session_state["active_dataset"]    = uploaded.name
    st.session_state["dataset_row_count"] = len(df)
    st.session_state["dataset_col_count"] = len(df.columns)
    st.success(
        f"Loaded **{uploaded.name}** — "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    # ── Target column ──────────────────────────────────────
    st.markdown('<div class="section-header">Select target column</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. MLForge requires a numeric regression target.")
        st.stop()

    target_col = st.selectbox(
        "Column to predict (y)",
        options=numeric_cols,
        help="All other numeric columns become features (X).",
    )
    st.session_state["target_col"] = target_col

    # ── Preview ────────────────────────────────────────────
    st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
    col_prev, col_stats = st.columns([3, 1])

    with col_prev:
        st.dataframe(df.head(8), width='stretch')

    with col_stats:
        missing     = df.isnull().sum().sum()
        missing_pct = 100 * missing / (df.shape[0] * df.shape[1])
        st.markdown("**Missing values**")
        if missing == 0:
            st.markdown('<span class="badge-good">None</span>', unsafe_allow_html=True)
        elif missing_pct < 5:
            st.markdown(f'<span class="badge-warn">{missing} ({missing_pct:.1f}%)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-poor">{missing} ({missing_pct:.1f}%)</span>', unsafe_allow_html=True)
        st.markdown("**Target**")
        st.code(target_col, language=None)
        st.markdown("**Target stats**")
        st.dataframe(df[target_col].describe().rename("value").to_frame(), width='stretch')

    # ── Dataset profiling ──────────────────────────────────
    st.markdown('<div class="section-header">Dataset profile</div>', unsafe_allow_html=True)
    with st.spinner("Profiling dataset…"):
        profile = profile_dataset(df, target_col)
        st.session_state["dataset_profile"] = profile

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Rows",         f"{profile['n_rows']:,}")
    pc2.metric("Features",     profile["n_features"])
    pc3.metric("Missing %",    f"{profile['missing_pct']:.1f}%")
    pc4.metric("Time-series?", "Yes" if profile["is_time_series"] else "No")

    for w in profile.get("warnings", []):
        st.warning(w)

    with st.expander("Full feature statistics"):
        st.dataframe(profile["feature_stats"], width='stretch')

    st.info(
        "Dataset ready. Head to **Train & Diagnostics** in the sidebar to start.",
        icon="✅",
    )
