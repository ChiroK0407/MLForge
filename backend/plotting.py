# backend/plotting.py
# ─────────────────────────────────────────────────────────────
# All plotting functions used across pages.
# Every function returns a matplotlib Figure so pages can call
# st.pyplot(fig) or save fig to disk for report embedding.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Shared style ───────────────────────────────────────────
PALETTE = ["#4f86c6", "#e06c5a", "#5ab58a", "#e0a84a",
           "#9b7dd4", "#5bbfcc", "#d47d9b", "#7d9b5b"]

def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent minimal styling to an axes."""
    ax.set_title(title, fontsize=11, fontweight="500", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)


# ── 1. Actual vs Predicted line plot ──────────────────────

def plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_col: str = "target",
    model_label: str = "Model",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(y_test))
    ax.plot(x, y_test, label="Actual",    color=PALETTE[0], linewidth=1.4)
    ax.plot(x, y_pred, label=model_label, color=PALETTE[1], linewidth=1.2, alpha=0.85)
    ax.legend(fontsize=8, framealpha=0.4)
    _style_ax(ax, f"Actual vs Predicted — {target_col}", "Sample index", target_col)
    fig.tight_layout()
    return fig


# ── 2. Multi-model overlay line plot ──────────────────────

def plot_multi_overlay(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    target_col: str = "target",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(y_test))
    ax.plot(x, y_test, label="Actual", color="black", linewidth=1.6, zorder=5)
    for i, (label, y_pred) in enumerate(predictions_dict.items()):
        ax.plot(x, y_pred, label=label, color=PALETTE[i % len(PALETTE)],
                linewidth=1.1, alpha=0.8)
    ax.legend(fontsize=8, framealpha=0.4, ncol=2)
    _style_ax(ax, f"Model comparison — {target_col}", "Sample index", target_col)
    fig.tight_layout()
    return fig


# ── 3. Scatter: actual vs predicted ───────────────────────

def plot_scatter(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_col: str = "target",
    model_label: str = "Model",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.7, label="Ideal")
    ax.scatter(y_test, y_pred, alpha=0.55, s=18, color=PALETTE[0], label=model_label)
    ax.legend(fontsize=8, framealpha=0.4)
    _style_ax(ax, f"Actual vs Predicted scatter", f"Actual {target_col}", "Predicted")
    fig.tight_layout()
    return fig


# ── 4. Multi-model scatter ────────────────────────────────

def plot_multi_scatter(
    y_test: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    target_col: str = "target",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = y_test.min()
    hi = y_test.max()
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.6, label="Ideal")
    for i, (label, y_pred) in enumerate(predictions_dict.items()):
        ax.scatter(y_test, y_pred, alpha=0.5, s=14,
                   color=PALETTE[i % len(PALETTE)], label=label)
    ax.legend(fontsize=8, framealpha=0.4, ncol=2)
    _style_ax(ax, "Actual vs Predicted — all models", f"Actual {target_col}", "Predicted")
    fig.tight_layout()
    return fig


# ── 5. Residuals plot ─────────────────────────────────────

def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_col: str = "target",
) -> plt.Figure:
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: residuals over index
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].scatter(np.arange(len(residuals)), residuals,
                    alpha=0.5, s=12, color=PALETTE[0])
    _style_ax(axes[0], "Residuals over samples", "Sample index", "Residual")

    # Right: residual distribution
    axes[1].hist(residuals, bins=30, color=PALETTE[0], alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="gray", linewidth=0.8, linestyle="--")
    _style_ax(axes[1], "Residual distribution", "Residual", "Count")

    fig.tight_layout()
    return fig


# ── 6. Feature importance bar chart ──────────────────────

def plot_feature_importance(
    df_imp: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Parameters
    ----------
    df_imp : DataFrame with columns [rank, feature, importance_pct]
    top_n  : show at most this many features
    """
    df_plot = df_imp.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(df_plot))))
    bars = ax.barh(df_plot["feature"][::-1], df_plot["importance_pct"][::-1],
                   color=PALETTE[0], alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=7.5, padding=3)
    ax.set_xlim(0, df_plot["importance_pct"].max() * 1.2)
    _style_ax(ax, f"Feature importance (top {len(df_plot)})", "Importance (%)", "")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ── 7. Metrics comparison bar chart ──────────────────────

def plot_metrics_comparison(
    metrics_dict: dict[str, dict],
    metric: str = "R2",
) -> plt.Figure:
    """
    Parameters
    ----------
    metrics_dict : {model_label: {MSE, RMSE, MAE, R2}}
    metric       : which metric to plot
    """
    labels = list(metrics_dict.keys())
    values = [metrics_dict[l].get(metric, 0) for l in labels]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels)), 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    _style_ax(ax, f"{metric} comparison across models", "Model", metric)
    fig.tight_layout()
    return fig


# ── 8. Autotune trial history ─────────────────────────────

def plot_autotune_history(cv_results: pd.DataFrame) -> plt.Figure:
    """
    Plot R² score across Optuna trials.
    cv_results must have a 'cv_r2_mean' column.
    """
    if "cv_r2_mean" not in cv_results.columns or cv_results.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No trial data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(8, 3))
    scores = cv_results["cv_r2_mean"].values
    ax.plot(scores, color=PALETTE[0], linewidth=1.2, alpha=0.7, label="Trial R²")
    # running best
    running_best = np.maximum.accumulate(scores)
    ax.plot(running_best, color=PALETTE[1], linewidth=1.6, label="Best so far")
    ax.legend(fontsize=8, framealpha=0.4)
    _style_ax(ax, "Auto-tune trial history", "Trial #", "CV R²")
    fig.tight_layout()
    return fig


# ── 9. Correlation heatmap ────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 12,
) -> plt.Figure:
    """Show correlation of top_n features with the target."""
    numeric_df = df.select_dtypes(include="number")
    if target_col not in numeric_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Target not numeric", ha="center", va="center")
        return fig

    corr_with_target = numeric_df.corr()[target_col].drop(target_col)
    top_features = corr_with_target.abs().nlargest(top_n).index
    subset = numeric_df[[target_col] + list(top_features)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        subset.corr(),
        ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cmap="coolwarm", center=0,
        linewidths=0.4, linecolor="white",
        square=True, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(f"Correlation heatmap (top {top_n} features)", fontsize=11, pad=10)
    ax.tick_params(labelsize=7.5, rotation=30)
    fig.tight_layout()
    return fig
