# config/param_spaces.py
# ─────────────────────────────────────────────────────────────
# Hyperparameter search spaces for every model in MODEL_REGISTRY.
# Used by backend/autotune.py (Optuna + RandomizedSearchCV).
#
# Format per param:
#   "type"    : "float" | "int" | "categorical"
#   "low"     : lower bound      (float / int)
#   "high"    : upper bound      (float / int)
#   "log"     : bool             (log-uniform sampling)
#   "choices" : list             (categorical)
# ─────────────────────────────────────────────────────────────

PARAM_SPACES: dict[str, dict] = {

    # ── SVR ───────────────────────────────────────────────
    "svr_linear": {
        "C":       {"type": "float", "low": 0.01,  "high": 200.0, "log": True},
        "epsilon": {"type": "float", "low": 0.001, "high": 1.0,   "log": True},
    },
    "svr_poly": {
        "C":       {"type": "float", "low": 0.01,  "high": 100.0, "log": True},
        "epsilon": {"type": "float", "low": 0.001, "high": 1.0,   "log": True},
        "degree":  {"type": "int",   "low": 2,     "high": 4,     "log": False},
        "coef0":   {"type": "float", "low": 0.0,   "high": 10.0,  "log": False},
    },
    "svr_rbf": {
        "C":       {"type": "float", "low": 0.01,  "high": 200.0, "log": True},
        "epsilon": {"type": "float", "low": 0.001, "high": 1.0,   "log": True},
        "gamma":   {"type": "float", "low": 1e-4,  "high": 10.0,  "log": True},
    },

    # ── Tree-Based ────────────────────────────────────────
    "decision_tree": {
        "max_depth":        {"type": "int",         "low": 2,    "high": 20,   "log": False},
        "min_samples_split":{"type": "int",         "low": 2,    "high": 20,   "log": False},
        "min_samples_leaf": {"type": "int",         "low": 1,    "high": 10,   "log": False},
        "max_features":     {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },

    # ── Ensemble — Bagging family ─────────────────────────
    "rf": {
        "n_estimators":      {"type": "int",         "low": 50,  "high": 500,  "log": False},
        "max_depth":         {"type": "int",         "low": 3,   "high": 30,   "log": False},
        "min_samples_split": {"type": "int",         "low": 2,   "high": 20,   "log": False},
        "min_samples_leaf":  {"type": "int",         "low": 1,   "high": 10,   "log": False},
        "max_features":      {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "extra_trees": {
        "n_estimators":      {"type": "int",         "low": 50,  "high": 500,  "log": False},
        "max_depth":         {"type": "int",         "low": 3,   "high": 30,   "log": False},
        "min_samples_split": {"type": "int",         "low": 2,   "high": 20,   "log": False},
        "max_features":      {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "bagging": {
        "n_estimators":  {"type": "int",   "low": 5,  "high": 100, "log": False},
        "max_samples":   {"type": "float", "low": 0.5,"high": 1.0, "log": False},
        "max_features":  {"type": "float", "low": 0.5,"high": 1.0, "log": False},
    },

    # ── Boosting family ───────────────────────────────────
    "xgb": {
        "learning_rate":       {"type": "float", "low": 0.01, "high": 0.3,   "log": True},
        "max_iter":            {"type": "int",   "low": 100,  "high": 1000,  "log": False},
        "max_depth":           {"type": "int",   "low": 3,    "high": 10,    "log": False},
        "min_samples_leaf":    {"type": "int",   "low": 1,    "high": 50,    "log": False},
        "l2_regularization":   {"type": "float", "low": 0.0,  "high": 10.0,  "log": False},
    },
    "gradient_boosting": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500,  "log": False},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
        "max_depth":       {"type": "int",   "low": 2,    "high": 8,    "log": False},
        "min_samples_split":{"type": "int",  "low": 2,    "high": 20,   "log": False},
        "subsample":       {"type": "float", "low": 0.5,  "high": 1.0,  "log": False},
    },
    "adaboost": {
        "n_estimators":  {"type": "int",   "low": 20,   "high": 300,  "log": False},
        "learning_rate": {"type": "float", "low": 0.01, "high": 2.0,  "log": True},
    },
    "lightgbm": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500,  "log": False},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
        "max_depth":       {"type": "int",   "low": 3,    "high": 12,   "log": False},
        "num_leaves":      {"type": "int",   "low": 15,   "high": 127,  "log": False},
        "min_child_samples":{"type": "int",  "low": 5,    "high": 100,  "log": False},
        "reg_alpha":       {"type": "float", "low": 0.0,  "high": 5.0,  "log": False},
        "reg_lambda":      {"type": "float", "low": 0.0,  "high": 5.0,  "log": False},
    },
    "catboost": {
        "iterations":    {"type": "int",   "low": 100,  "high": 1000, "log": False},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
        "depth":         {"type": "int",   "low": 3,    "high": 10,   "log": False},
        "l2_leaf_reg":   {"type": "float", "low": 0.5,  "high": 10.0, "log": True},
    },

    # ── Linear models ─────────────────────────────────────
    "linear_reg": {
        # No meaningful hyperparameters to tune for OLS
        "fit_intercept": {"type": "categorical", "choices": [True]},
    },
    "ridge": {
        "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
    },
    "elasticnet": {
        "alpha":    {"type": "float", "low": 0.001, "high": 10.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.0,   "high": 1.0,  "log": False},
    },
    "poly_reg": {
        "degree": {"type": "int", "low": 2, "high": 4, "log": False},
    },
    "bayesian_ridge": {
        "alpha_1": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
        "alpha_2": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
        "lambda_1":{"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
        "lambda_2":{"type": "float", "low": 1e-7, "high": 1e-3, "log": True},
    },
    "kernel_ridge": {
        "alpha":  {"type": "float",      "low": 0.001, "high": 100.0, "log": True},
        "kernel": {"type": "categorical","choices": ["rbf", "poly", "linear"]},
        "gamma":  {"type": "float",      "low": 1e-4,  "high": 10.0,  "log": True},
    },

    # ── Neural Network ────────────────────────────────────
    "mlp": {
        "hidden_layer_sizes": {
            "type": "categorical",
            "choices": [
                (50,), (100,), (200,),
                (100, 50), (100, 100), (200, 100), (200, 100, 50),
            ],
        },
        "alpha":             {"type": "float", "low": 1e-5, "high": 0.1,  "log": True},
        "learning_rate_init":{"type": "float", "low": 1e-4, "high": 0.01, "log": True},
        "max_iter":          {"type": "int",   "low": 500,  "high": 3000, "log": False},
    },

    # ── Probabilistic ─────────────────────────────────────
    "gaussian_process": {
        # GP hyperparameters are learned internally via marginal likelihood.
        # The main tuning lever exposed to users is the noise level (alpha).
        "alpha":         {"type": "float", "low": 1e-5, "high": 1.0,  "log": True},
        "n_restarts_optimizer": {"type": "int", "low": 0, "high": 5, "log": False},
    },

    # ── Other ─────────────────────────────────────────────
    "knn": {
        "n_neighbors": {"type": "int",         "low": 2,  "high": 30, "log": False},
        "weights":     {"type": "categorical", "choices": ["uniform", "distance"]},
        "p":           {"type": "int",         "low": 1,  "high": 2,  "log": False},
    },
}


def get_space(model_key: str) -> dict:
    """Return the param space for a model key. Raises KeyError if missing."""
    if model_key not in PARAM_SPACES:
        raise KeyError(
            f"No param space defined for '{model_key}'. "
            f"Available: {list(PARAM_SPACES.keys())}"
        )
    return PARAM_SPACES[model_key]


def space_summary(model_key: str) -> list[str]:
    """
    Human-readable list of param ranges shown in the UI.
    e.g. ["C : log-uniform [0.01, 200.0]", "epsilon : log-uniform [0.001, 1.0]"]
    """
    lines = []
    for param, cfg in PARAM_SPACES.get(model_key, {}).items():
        if cfg["type"] == "categorical":
            lines.append(f"{param} : categorical {cfg['choices']}")
        else:
            scale = "log-uniform" if cfg.get("log") else "uniform"
            lines.append(f"{param} : {scale} [{cfg['low']}, {cfg['high']}]")
    return lines


def has_space(model_key: str) -> bool:
    """Return True if a param space exists for this model."""
    return model_key in PARAM_SPACES
