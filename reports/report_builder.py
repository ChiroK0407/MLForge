# reports/report_builder.py
# ─────────────────────────────────────────────────────────────
# Assembles a format-agnostic report payload from session runs.
# v3: Uses pre-stored artifacts from run["artifacts"] where
#     available. Regenerates actual_vs_predicted and residuals
#     from y_test/y_pred only as fallback. Feature importance
#     and autotune artifacts are taken from store only —
#     there is no regeneration path for those.
# ─────────────────────────────────────────────────────────────

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.plotting       import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_metrics_comparison,
)
from backend.analyze_helper import interpret_metrics
from config.model_registry  import MODEL_REGISTRY


def _fig_to_buf(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _bytes_to_buf(b: bytes) -> io.BytesIO:
    """Wrap raw PNG bytes in a seeked BytesIO for python-docx / base64."""
    buf = io.BytesIO(b)
    buf.seek(0)
    return buf


# ── Dynamic reference builder ──────────────────────────────

def build_references(runs: list[dict], dataset_name: str, autotune_used: bool) -> list[tuple]:
    used_keys = {r.get("model_key", "") for r in runs}
    ds_lower  = (dataset_name or "").lower()
    refs      = []

    refs.append((
        "Pedregosa et al., 2011",
        "Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, 12, 2825–2830.",
    ))

    if autotune_used:
        refs.append((
            "Akiba et al., 2019",
            "Optuna: A Next-generation Hyperparameter Optimization Framework. "
            "Proceedings of the 25th ACM SIGKDD International Conference on "
            "Knowledge Discovery & Data Mining.",
        ))

    if used_keys & {"svr_linear", "svr_poly", "svr_rbf"}:
        refs.append(("Cortes & Vapnik, 1995",
                      "Support-Vector Networks. Machine Learning, 20(3), 273–297."))
        refs.append(("Smola & Schölkopf, 2004",
                      "A tutorial on support vector regression. "
                      "Statistics and Computing, 14(3), 199–222."))

    if used_keys & {"rf", "extra_trees", "bagging"}:
        refs.append(("Breiman, 2001", "Random Forests. Machine Learning, 45(1), 5–32."))

    if used_keys & {"xgb", "gradient_boosting", "adaboost"}:
        refs.append(("Friedman, 2001",
                      "Greedy function approximation: A gradient boosting machine. "
                      "Annals of Statistics, 29(5), 1189–1232."))

    if "lightgbm" in used_keys:
        refs.append(("Ke et al., 2017",
                      "LightGBM: A Highly Efficient Gradient Boosting Decision Tree. "
                      "Advances in Neural Information Processing Systems, 30."))

    if "catboost" in used_keys:
        refs.append(("Prokhorenkova et al., 2018",
                      "CatBoost: unbiased boosting with categorical features. "
                      "Advances in Neural Information Processing Systems, 31."))

    if "mlp" in used_keys:
        refs.append(("Bishop, 2006",
                      "Pattern Recognition and Machine Learning. Springer."))

    indpensim_signals = ["indpensim", "penicillin", "ferment", "pencon", "pen_"]
    if any(sig in ds_lower for sig in indpensim_signals):
        refs.append(("Goldrick et al., 2019",
                      "Modern day monitoring and control challenges outlined on an "
                      "industrial-scale benchmark fermentation process. "
                      "Computers & Chemical Engineering, 130, 106471. "
                      "doi:10.1016/j.compchemeng.2019.05.037"))

    return refs


# ── Main payload builder ───────────────────────────────────

def build_report_payload(
    runs:            list[dict],
    report_cfg:      dict,
    dataset_profile: dict,
    ai_sections:     dict | None = None,
    per_run_include: dict | None = None,
) -> dict:
    """
    Assemble everything needed by any exporter into one dict.

    Parameters
    ----------
    runs            : list of saved run dicts from session_store
    report_cfg      : {title, author, institution, date, notes, include_plots}
    dataset_profile : output of backend.dataset_profiler.profile_dataset()
    ai_sections     : {conclusions, research_gaps, future_work}
    per_run_include : {run_id: [artifact_key, ...]}
                      Which artifacts to include for each run.
                      If None, all stored artifacts are included.

    Returns
    -------
    payload dict — keys: meta, profile, runs (enriched), comparison_rows,
                         comp_plot_buf, best_run, has_plots, n_runs,
                         references, ai_sections
    """
    include_plots  = report_cfg.get("include_plots", True)
    enriched_runs  = []
    autotune_used  = False

    for run in runs:
        metrics    = run.get("metrics",    {})
        model_key  = run.get("model_key",  "")
        y_test     = run.get("y_test")
        y_pred     = run.get("y_pred")
        target_col = run.get("target_col", "target")
        run_id     = run.get("run_id",     0)
        stored     = run.get("artifacts",  {})

        # Which artifacts to include for this run
        include_keys = (
            set(per_run_include.get(run_id, list(stored.keys())))
            if per_run_include
            else set(stored.keys())
        )

        if run.get("best_params"):
            autotune_used = True

        analysis  = interpret_metrics(metrics, model_key, {}, target_col)
        plot_bufs = {}

        if include_plots:
            # ── Actual vs Predicted ───────────────────────
            # Use stored bytes if saved, else regenerate from y_test/y_pred
            if "actual_vs_predicted" in include_keys:
                if "actual_vs_predicted" in stored:
                    plot_bufs["actual_vs_predicted"] = _bytes_to_buf(
                        stored["actual_vs_predicted"]
                    )
                elif y_test is not None and y_pred is not None:
                    try:
                        fig = plot_actual_vs_predicted(
                            y_test, y_pred, target_col,
                            MODEL_REGISTRY.get(model_key, {}).get("label", model_key),
                        )
                        plot_bufs["actual_vs_predicted"] = _fig_to_buf(fig)
                    except Exception:
                        pass

            # ── Residuals ─────────────────────────────────
            if "residuals" in include_keys:
                if "residuals" in stored:
                    plot_bufs["residuals"] = _bytes_to_buf(stored["residuals"])
                elif y_test is not None and y_pred is not None:
                    try:
                        fig = plot_residuals(y_test, y_pred, target_col)
                        plot_bufs["residuals"] = _fig_to_buf(fig)
                    except Exception:
                        pass

            # ── Feature importance ── stored only, no regeneration ──
            if "feature_importance" in include_keys and "feature_importance" in stored:
                plot_bufs["feature_importance"] = _bytes_to_buf(
                    stored["feature_importance"]
                )

            # ── Autotune history chart ─────────────────────
            if "autotune_history" in include_keys and "autotune_history" in stored:
                plot_bufs["autotune_history"] = _bytes_to_buf(
                    stored["autotune_history"]
                )

        # ── Autotune text artifacts ────────────────────────
        autotune_metrics = None
        if "autotune_metrics" in include_keys and "autotune_metrics" in stored:
            autotune_metrics = stored["autotune_metrics"]

        autotune_comparison = None
        if "autotune_comparison" in include_keys and "autotune_comparison" in stored:
            autotune_comparison = stored["autotune_comparison"]

        enriched_runs.append({
            **run,
            "analysis":           analysis,
            "plot_bufs":          plot_bufs,
            "autotune_metrics":   autotune_metrics,
            "autotune_comparison": autotune_comparison,
        })

    # ── Comparison table ───────────────────────────────────
    comparison_rows = []
    for i, run in enumerate(enriched_runs):
        m = run.get("metrics", {})
        comparison_rows.append({
            "Run #":  run.get("run_id", i + 1),
            "Model":  run.get("model_label", "—"),
            "R²":     round(m.get("R2",   0), 4),
            "RMSE":   round(m.get("RMSE", 0), 4),
            "MAE":    round(m.get("MAE",  0), 4),
            "MSE":    round(m.get("MSE",  0), 6),
            "Notes":  run.get("notes", ""),
        })

    # ── Comparison bar chart ───────────────────────────────
    comp_plot_buf = None
    if include_plots and len(enriched_runs) > 1:
        try:
            metrics_dict = {
                f"#{r['run_id']} {r['model_label']}": r["metrics"]
                for r in enriched_runs
            }
            fig_cmp = plot_metrics_comparison(metrics_dict, metric="R2")
            comp_plot_buf = _fig_to_buf(fig_cmp)
        except Exception:
            pass

    # ── Best run ───────────────────────────────────────────
    best_run = max(
        enriched_runs,
        key=lambda r: r.get("metrics", {}).get("R2", -999),
        default=None,
    )

    # ── Dynamic references ─────────────────────────────────
    dataset_name = runs[0].get("dataset_name", "") if runs else ""
    references   = build_references(runs, dataset_name, autotune_used)

    # ── AI sections fallback ───────────────────────────────
    if not ai_sections or not any(ai_sections.values()):
        from reports.ai_conclusions import _fallback
        ai_sections = _fallback(enriched_runs, dataset_profile)

    return {
        "meta":             report_cfg,
        "profile":          dataset_profile,
        "runs":             enriched_runs,
        "comparison_rows":  comparison_rows,
        "comp_plot_buf":    comp_plot_buf,
        "best_run":         best_run,
        "has_plots":        include_plots,
        "n_runs":           len(enriched_runs),
        "references":       references,
        "ai_sections":      ai_sections,
    }
