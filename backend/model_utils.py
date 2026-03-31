# backend/model_utils.py
# ─────────────────────────────────────────────────────────────
# Core preprocessing and training pipeline.
# All hardcoded test_size=0.2 values replaced by cfg dict.
# Supports every model in config/model_registry.py.
# ─────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from config.model_registry import MODEL_REGISTRY, build_model


# ── Preprocessing ──────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    target_col: str,
    sort_by_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, SimpleImputer, list[str]]:
    """
    Robust preprocessing for any tabular dataset.

    Steps
    -----
    1. Drop rows where target is missing
    2. Sort by 'time' column if present and sort_by_time=True
    3. Parse datetime/object columns → numeric hour/day/month features
    4. Keep only numeric columns
    5. Impute missing values (column mean)
    6. StandardScale features

    Returns
    -------
    X_scaled, y, scaler, imputer, feature_names
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    if sort_by_time:
        time_candidates = [c for c in df.columns if c.lower() in ("time", "t")]
        if time_candidates:
            df = df.sort_values(by=time_candidates[0]).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(float)

    # Parse datetime / object columns
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col + "_hour"]  = X[col].dt.hour
            X[col + "_day"]   = X[col].dt.day
            X[col + "_month"] = X[col].dt.month
            X = X.drop(columns=[col])
        elif X[col].dtype == "object":
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(parsed):
                X[col + "_hour"]  = parsed.dt.hour
                X[col + "_day"]   = parsed.dt.day
                X[col + "_month"] = parsed.dt.month
            X = X.drop(columns=[col])

    # Keep numeric only
    X = X.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    feature_names = list(X.columns)

    # Impute + scale
    imputer = SimpleImputer(strategy="mean")
    X_imp   = imputer.fit_transform(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, y, scaler, imputer, feature_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X, y according to cfg["split_strategy"] and cfg["test_size"].

    Strategies
    ----------
    "Time-ordered split"   : sequential slice (no shuffle)
    "Random shuffle split" : sklearn train_test_split with random_state
    """
    test_size = cfg.get("test_size", 0.2)

    if cfg.get("split_strategy", "Time-ordered split") == "Time-ordered split":
        n_train = int((1.0 - test_size) * len(X))
        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    else:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=cfg.get("random_seed", 42),
        )


# ── Metrics ────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MSE, RMSE, MAE, R² as a flat dict."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MSE":  round(mse, 6),
        "RMSE": round(float(np.sqrt(mse)), 6),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 6),
        "R2":   round(float(r2_score(y_true, y_pred)), 6),
    }


# ── Main training function ─────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_key: str,
    cfg: dict,
    custom_params: dict | None = None,
) -> dict:
    """
    Full pipeline: preprocess → split → train → evaluate.

    Parameters
    ----------
    df           : raw dataframe
    target_col   : regression target column name
    model_key    : key from MODEL_REGISTRY
    cfg          : sidebar config dict (train_size, split_strategy, etc.)
    custom_params: override model default params (e.g. from autotune)

    Returns
    -------
    dict with keys:
        model, metrics, data, scaler, imputer, feature_names,
        model_key, cfg_snapshot
    """
    sort_by_time = "time" in [c.lower() for c in df.columns]

    X_scaled, y, scaler, imputer, feature_names = preprocess(
        df, target_col, sort_by_time=sort_by_time
    )

    X_train, X_test, y_train, y_test = split_data(X_scaled, y, cfg)

    model = build_model(model_key, custom_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    metrics  = compute_metrics(y_test, y_pred)

    return {
        "model":         model,
        "metrics":       metrics,
        "data": {
            "X_train":        X_train,
            "X_test":         X_test,
            "y_train":        y_train,
            "y_test":         y_test,
            "y_pred":         y_pred,
            # DataFrame versions for plotting (preserves feature names)
            "X_train_df":     pd.DataFrame(X_train, columns=feature_names),
            "X_test_df":      pd.DataFrame(X_test,  columns=feature_names),
        },
        "scaler":        scaler,
        "imputer":       imputer,
        "feature_names": feature_names,
        "model_key":     model_key,
        "cfg_snapshot":  cfg.copy(),
    }


def train_multiple_models(
    df: pd.DataFrame,
    target_col: str,
    model_keys: list[str],
    cfg: dict,
) -> dict[str, dict]:
    """
    Train several models on the same preprocessed split.

    Returns
    -------
    {model_key: train_model() result dict}
    """
    # Preprocess once, reuse split for a fair comparison
    sort_by_time = "time" in [c.lower() for c in df.columns]

    X_scaled, y, scaler, imputer, feature_names = preprocess(
        df, target_col, sort_by_time=sort_by_time
    )
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, cfg)

    results = {}
    for key in model_keys:
        try:
            model = build_model(key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_train, y_train)

            y_pred  = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            results[key] = {
                "model":         model,
                "metrics":       metrics,
                "data": {
                    "X_train":    X_train,
                    "X_test":     X_test,
                    "y_train":    y_train,
                    "y_test":     y_test,
                    "y_pred":     y_pred,
                    "X_train_df": pd.DataFrame(X_train, columns=feature_names),
                    "X_test_df":  pd.DataFrame(X_test,  columns=feature_names),
                },
                "scaler":        scaler,
                "imputer":       imputer,
                "feature_names": feature_names,
                "model_key":     key,
                "cfg_snapshot":  cfg.copy(),
            }
        except Exception as e:
            results[key] = {"error": str(e), "model_key": key}

    return results


# ── Feature importance ─────────────────────────────────────

def extract_feature_importance(
    model,
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """
    Extract normalised feature importance from a trained model.

    Falls back to correlation-based importance for models without
    native coef_ or feature_importances_ attributes (e.g. KNN, RBF SVR).

    Returns
    -------
    DataFrame with columns: rank, feature, importance_pct
    """
    importances = None

    if hasattr(model, "coef_") and model.coef_ is not None:
        importances = np.abs(np.array(model.coef_).flatten())

    elif hasattr(model, "w") and model.w is not None:
        # scratch linear SVR
        importances = np.abs(model.w)

    elif hasattr(model, "feature_importances_"):
        importances = np.abs(model.feature_importances_)

    if importances is None or len(importances) != len(feature_names):
        # fallback: Pearson correlation with target
        corr = np.array([
            abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
            for i in range(X_train.shape[1])
        ])
        importances = np.nan_to_num(corr)

    total = importances.sum()
    if total > 0:
        pct = 100.0 * importances / total
    else:
        pct = np.zeros_like(importances)

    df_imp = pd.DataFrame({
        "feature":        feature_names,
        "importance_pct": np.round(pct, 2),
    }).sort_values("importance_pct", ascending=False).reset_index(drop=True)

    df_imp.insert(0, "rank", np.arange(1, len(df_imp) + 1))
    return df_imp


# ── Retrain on subset of features ─────────────────────────

def retrain_reduced(
    df: pd.DataFrame,
    target_col: str,
    model_key: str,
    cfg: dict,
    selected_features: list[str],
    custom_params: dict | None = None,
) -> dict:
    """
    Retrain using only selected_features columns.
    Returns same structure as train_model() plus rmse key.
    """
    cols = selected_features + [target_col]
    result = train_model(df[cols], target_col, model_key, cfg, custom_params)
    result["rmse"] = float(np.sqrt(result["metrics"]["MSE"]))
    return result
