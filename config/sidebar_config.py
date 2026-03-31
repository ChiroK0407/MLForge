# config/sidebar_config.py
# ─────────────────────────────────────────────────────────────
# v3 — Sidebar is now a STATUS PANEL only.
# No training controls live here anymore.
#
# Controls moved to pages:
#   Train/test split, split strategy, random seed → Page 1
#   CV folds, autotune budget                    → Page 1 (autotune expander)
#   Hide slow models toggle                      → inline on each page
#
# Pages call get_train_cfg() to read the current config from
# session state. Page 1 writes it. All other pages read it.
# ─────────────────────────────────────────────────────────────

import streamlit as st
from config.model_registry import MODEL_REGISTRY


# ── Defaults ───────────────────────────────────────────────
DEFAULT_CFG = {
    "train_size":      0.80,
    "test_size":       0.20,
    "k_folds":         3,
    "split_strategy":  "Time-ordered split",
    "random_seed":     42,
    "autotune_budget": 20,
    "hide_slow":       False,
}


def get_train_cfg() -> dict:
    """
    Return the current training config from session state.
    Falls back to DEFAULT_CFG if Page 1 has never been visited.
    All pages should call this instead of reading from the sidebar.
    """
    return st.session_state.get("train_cfg", DEFAULT_CFG.copy())


def save_train_cfg(cfg: dict) -> None:
    """Persist config dict to session state. Called by Page 1."""
    st.session_state["train_cfg"] = cfg


def render_sidebar() -> dict:
    """
    Draw the status-only sidebar.
    Returns get_train_cfg() so pages can still use cfg["key"] as before.
    """
    with st.sidebar:

        # ── Dataset status ────────────────────────────────
        st.markdown("##### Dataset")
        active = st.session_state.get("active_dataset", None)
        if active:
            st.success(f"📂 {active}", icon=None)
            row_count = st.session_state.get("dataset_row_count", "—")
            col_count = st.session_state.get("dataset_col_count", "—")
            target    = st.session_state.get("target_col", "—")
            st.caption(f"{row_count:,} rows · {col_count} cols" if isinstance(row_count, int) else f"{row_count} rows · {col_count} cols")
            st.caption(f"Target: **{target}**")
        else:
            st.info("No dataset loaded.\nUpload on Home page.", icon=None)

        st.divider()

        # ── Current config (read-only summary) ────────────
        cfg = get_train_cfg()
        st.markdown("##### Training config")
        st.caption(
            f"Split: {cfg['split_strategy'].split()[0]}  "
            f"· Train: {int(cfg['train_size'] * 100)}%  "
            f"· Seed: {cfg['random_seed']}"
        )
        st.caption(
            f"CV folds: {cfg['k_folds']}  "
            f"· Optuna trials: {cfg['autotune_budget']}"
        )
        st.caption("_Set on Train & Diagnostics page_")

        st.divider()

        # ── Run history ───────────────────────────────────
        runs = st.session_state.get("runs", [])
        if runs:
            st.markdown(f"##### Saved runs ({len(runs)})")
            for i, run in enumerate(runs[-4:]):
                model_label = MODEL_REGISTRY.get(
                    run.get("model_key", ""), {}
                ).get("label", run.get("model_key", "—"))
                r2 = run.get("metrics", {}).get("R2", None)
                r2_str = f"R²={r2:.3f}" if r2 is not None else ""
                run_num = len(runs) - len(runs[-4:]) + i + 1
                st.caption(f"#{run_num}  {model_label}  {r2_str}")
            if len(runs) > 4:
                st.caption(f"… and {len(runs) - 4} more on Page 5")
        else:
            st.caption("No runs saved yet.")

        st.divider()

        # ── Reset ─────────────────────────────────────────
        if st.button("🗑  Reset everything", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return cfg


def apply_page_config(title: str = "MLForge") -> None:
    """
    Must be the very first Streamlit call on every page.
    """
    st.set_page_config(
        page_title=title,
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
