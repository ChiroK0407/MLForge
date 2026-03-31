# backend/autotune.py
# ─────────────────────────────────────────────────────────────
# Optuna-powered hyperparameter search.
# Falls back to RandomizedSearchCV if Optuna is unavailable.
#
# Usage:
#   from backend.autotune import run_autotune
#   result = run_autotune(model_key, X_train, y_train, cfg, budget=20)
# ─────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score
from sklearn.exceptions import ConvergenceWarning

from config.model_registry import build_model
from config.param_spaces   import get_space, PARAM_SPACES


# ── Optuna trial builder ───────────────────────────────────

def _suggest_params(trial, space: dict) -> dict:
    """
    Translate param_spaces.py spec into Optuna suggest_* calls.
    """
    params = {}
    for name, cfg in space.items():
        ptype = cfg["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
    return params


def _make_cv_splitter(cfg: dict):
    """Return a CV splitter consistent with sidebar config."""
    k = cfg.get("k_folds", 3)
    seed = cfg.get("random_seed", 42)
    shuffle = cfg.get("split_strategy", "Time-ordered split") != "Time-ordered split"
    return KFold(n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None)


# ── Main autotune entry point ──────────────────────────────

def run_autotune(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    budget: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Run hyperparameter search for model_key on (X_train, y_train).

    Parameters
    ----------
    model_key         : key from MODEL_REGISTRY / PARAM_SPACES
    X_train, y_train  : already-preprocessed training data
    cfg               : sidebar config dict
    budget            : number of Optuna trials (overrides cfg["autotune_budget"])
    progress_callback : optional callable(trial_num, total) for Streamlit progress bar

    Returns
    -------
    dict with keys:
        best_params   : dict of best hyperparameters found
        best_score    : float (mean CV R²)
        cv_results    : pd.DataFrame of all trials (params + score)
        n_trials      : int
        method        : "optuna" | "random_search" | "fixed_grid"
    """
    n_trials = budget or cfg.get("autotune_budget", 20)

    if model_key not in PARAM_SPACES:
        return {
            "best_params":  {},
            "best_score":   None,
            "cv_results":   pd.DataFrame(),
            "n_trials":     0,
            "method":       "none",
            "error":        f"No param space defined for '{model_key}'.",
        }

    space = get_space(model_key)
    cv    = _make_cv_splitter(cfg)

    # ── Try Optuna first ───────────────────────────────────
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        trial_records = []
        completed = [0]

        def objective(trial):
            params  = _suggest_params(trial, space)
            model   = build_model(model_key, params)
            scorer  = make_scorer(r2_score)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scorer, n_jobs=1,
                )

            mean_r2 = float(np.mean(scores))
            trial_records.append({**params, "cv_r2_mean": round(mean_r2, 5)})

            completed[0] += 1
            if progress_callback:
                progress_callback(completed[0], n_trials)

            return mean_r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score  = study.best_value
        cv_results  = pd.DataFrame(trial_records).sort_values(
            "cv_r2_mean", ascending=False
        ).reset_index(drop=True)

        return {
            "best_params": best_params,
            "best_score":  round(best_score, 5),
            "cv_results":  cv_results,
            "n_trials":    n_trials,
            "method":      "optuna",
        }

    except ImportError:
        pass  # fall through to RandomizedSearchCV

    # ── Fallback: RandomizedSearchCV ──────────────────────
    try:
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import loguniform, randint, uniform

        sklearn_space = _space_to_sklearn_distributions(space)
        base_model    = build_model(model_key)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            search = RandomizedSearchCV(
                base_model,
                param_distributions=sklearn_space,
                n_iter=n_trials,
                cv=cv,
                scoring="r2",
                random_state=cfg.get("random_seed", 42),
                n_jobs=1,
                refit=False,
            )
            search.fit(X_train, y_train)

        cv_df = pd.DataFrame(search.cv_results_)
        best_params = search.best_params_
        best_score  = float(search.best_score_)

        return {
            "best_params": best_params,
            "best_score":  round(best_score, 5),
            "cv_results":  cv_df[["params", "mean_test_score"]].rename(
                columns={"mean_test_score": "cv_r2_mean"}
            ),
            "n_trials":    n_trials,
            "method":      "random_search",
        }

    except Exception as e:
        return {
            "best_params":  {},
            "best_score":   None,
            "cv_results":   pd.DataFrame(),
            "n_trials":     0,
            "method":       "failed",
            "error":        str(e),
        }


def _space_to_sklearn_distributions(space: dict) -> dict:
    """
    Convert param_spaces spec to scipy distributions for
    RandomizedSearchCV fallback.
    """
    from scipy.stats import loguniform, randint, uniform

    dists = {}
    for name, cfg in space.items():
        ptype = cfg["type"]
        if ptype == "float":
            if cfg.get("log"):
                dists[name] = loguniform(cfg["low"], cfg["high"])
            else:
                dists[name] = uniform(cfg["low"], cfg["high"] - cfg["low"])
        elif ptype == "int":
            dists[name] = randint(cfg["low"], cfg["high"] + 1)
        elif ptype == "categorical":
            dists[name] = cfg["choices"]
    return dists


# ── Quick fixed-grid fallback (no scipy / optuna) ──────────

FIXED_GRIDS: dict[str, list[dict]] = {
    "svr_linear":  [{"C": 0.5, "epsilon": 0.01},
                    {"C": 1.0, "epsilon": 0.05},
                    {"C": 2.0, "epsilon": 0.1}],
    "svr_poly":    [{"C": 1.0, "epsilon": 0.05, "degree": 2},
                    {"C": 1.0, "epsilon": 0.05, "degree": 3},
                    {"C": 2.0, "epsilon": 0.1,  "degree": 2}],
    "svr_rbf":     [{"C": 1.0, "epsilon": 0.05, "gamma": "scale"},
                    {"C": 2.0, "epsilon": 0.1,  "gamma": 0.01},
                    {"C": 5.0, "epsilon": 0.1,  "gamma": 0.1}],
    "rf":          [{"n_estimators": 100}, {"n_estimators": 200},
                    {"n_estimators": 300, "max_depth": 10}],
    "ridge":       [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
}


def run_fixed_grid(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Minimal grid search using FIXED_GRIDS.
    Used as last-resort fallback or for scratch SVR models
    where scipy distributions aren't meaningful.
    """
    grid   = FIXED_GRIDS.get(model_key, [{}])
    cv     = _make_cv_splitter(cfg)
    scorer = make_scorer(r2_score)

    best_score  = -np.inf
    best_params = {}
    records     = []

    for params in grid:
        try:
            model = build_model(model_key, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                scores = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1
                )
            mean_r2 = float(np.mean(scores))
            records.append({**params, "cv_r2_mean": round(mean_r2, 5)})
            if mean_r2 > best_score:
                best_score  = mean_r2
                best_params = params
        except Exception:
            continue

    return {
        "best_params": best_params,
        "best_score":  round(best_score, 5),
        "cv_results":  pd.DataFrame(records),
        "n_trials":    len(grid),
        "method":      "fixed_grid",
    }
