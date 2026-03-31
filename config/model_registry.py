# config/model_registry.py
# ─────────────────────────────────────────────────────────────
# Single source of truth for every model in MLForge.
# Scratch SVR implementations removed --- this registry focuses
# on replicating the ML methods reported across hydrogen
# production / biofuel literature (papers_cleaned.csv).
#
# Coverage map (algorithm → registry key):
#   Random Forest        → rf
#   SVM                  → svr_linear, svr_poly, svr_rbf
#   XGBoost              → xgb
#   KNN                  → knn
#   Gradient Boosting    → xgb  (same backend)
#   Linear Regression    → linear_reg
#   Decision Tree        → decision_tree
#   ANN (shallow MLP)    → mlp
#   AdaBoost             → adaboost
#   Gaussian Process     → gaussian_process
#   LightGBM             → lightgbm
#   CatBoost             → catboost
#   Bagging              → bagging
#   Extra Trees          → extra_trees
#   Bayesian Ridge       → bayesian_ridge
#   Kernel Ridge         → kernel_ridge
#   Polynomial Regression→ poly_reg
#   Ridge                → ridge
#   ElasticNet           → elasticnet
# ─────────────────────────────────────────────────────────────

# ── sklearn imports ────────────────────────────────────────
from sklearn.svm              import SVR
from sklearn.ensemble         import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree             import DecisionTreeRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.linear_model     import (
    LinearRegression,
    Ridge,
    ElasticNet,
    BayesianRidge,
)
from sklearn.neighbors        import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge     import KernelRidge
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import PolynomialFeatures

# ── Optional heavy libraries (fail gracefully) ─────────────
try:
    from lightgbm import LGBMRegressor
    _LIGHTGBM_OK = True
except ImportError:
    _LIGHTGBM_OK = False

try:
    from catboost import CatBoostRegressor
    _CATBOOST_OK = True
except ImportError:
    _CATBOOST_OK = False


# ── Polynomial regression wrapper ─────────────────────────
# sklearn has no single PolynomialRegression class ---
# it's a Pipeline(PolynomialFeatures, LinearRegression).
# We expose it as a first-class model here.
class PolynomialRegression(Pipeline):
    """
    Thin wrapper around sklearn Pipeline so build_model()
    can instantiate it uniformly like any other model.
    """
    def __init__(self, degree: int = 2, fit_intercept: bool = True):
        self.degree        = degree
        self.fit_intercept = fit_intercept
        super().__init__(steps=[
            ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression(fit_intercept=fit_intercept)),
        ])


# ══════════════════════════════════════════════════════════
# GROUP CONSTANTS
# ══════════════════════════════════════════════════════════
GROUP_SVR           = "SVR"
GROUP_TREE          = "Tree-Based"
GROUP_ENSEMBLE      = "Ensemble"
GROUP_BOOST         = "Boosting"
GROUP_LINEAR        = "Linear"
GROUP_NEURAL        = "Neural Network"
GROUP_PROBABILISTIC = "Probabilistic"
GROUP_OTHER         = "Other"


# ══════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════
# Each entry:
#   label          : display name shown in the UI
#   cls            : model class (not instance)
#   group          : one of the GROUP_* constants above
#   default_params : constructor kwargs used for manual override UI
#                    AND as defaults when NOT autotuning.
#                    KEPT IN SYNC with param_spaces.py so every
#                    autotune-able param is also exposed as a manual
#                    control on Page 1.
#   description    : one-line tooltip
#   literature_name: how this algorithm appears in papers
#                    (used for paper-matching on Page 4)

MODEL_REGISTRY: dict[str, dict] = {

    # ══════════════════════════════════════════════════════
    # SVR  (3 sklearn kernels --- linear / poly / rbf)
    # Literature frequency: SVM 13x across 29 papers
    # ══════════════════════════════════════════════════════
    "svr_linear": {
        "label":           "SVR — Linear kernel",
        "cls":             SVR,
        "group":           GROUP_SVR,
        "default_params":  {
            "kernel":  "linear",
            "C":       1.0,
            "epsilon": 0.1,
        },
        "description":     "Support Vector Regression, linear kernel. Fast on large datasets.",
        "literature_name": "SVM",
    },

    "svr_poly": {
        "label":           "SVR — Polynomial kernel",
        "cls":             SVR,
        "group":           GROUP_SVR,
        "default_params":  {
            "kernel":  "poly",
            "C":       1.0,
            "epsilon": 0.1,
            "degree":  2,
            "coef0":   0.0,
        },
        "description":     "Support Vector Regression, polynomial kernel.",
        "literature_name": "SVM",
    },

    "svr_rbf": {
        "label":           "SVR — RBF kernel",
        "cls":             SVR,
        "group":           GROUP_SVR,
        "default_params":  {
            "kernel":  "rbf",
            "C":       1.0,
            "epsilon": 0.1,
            "gamma":   "scale",
        },
        "description":     "Support Vector Regression, RBF kernel. Best general-purpose SVR.",
        "literature_name": "SVM",
    },

    # ══════════════════════════════════════════════════════
    # Tree-Based
    # FIX: expanded default_params to expose all meaningful
    # hyperparameters as manual controls on Page 1, matching
    # what param_spaces.py already defines for autotune.
    # ══════════════════════════════════════════════════════
    "decision_tree": {
        "label":           "Decision Tree",
        "cls":             DecisionTreeRegressor,
        "group":           GROUP_TREE,
        "default_params":  {
            "max_depth":         None,   # None = fully grown; set an int to prune
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "max_features":      None,   # None = all features; "sqrt"/"log2" to subsample
            "random_state":      42,
        },
        "description":     "Single decision tree. Interpretable, prone to overfitting.",
        "literature_name": "Decision Tree",
    },

    # ══════════════════════════════════════════════════════
    # Ensemble --- Bagging family
    # Literature frequency: Random Forest 18x, Extra Trees 1x, Bagging 2x
    # FIX: expanded default_params for rf and extra_trees.
    # ══════════════════════════════════════════════════════
    "rf": {
        "label":           "Random Forest",
        "cls":             RandomForestRegressor,
        "group":           GROUP_ENSEMBLE,
        "default_params":  {
            "n_estimators":      100,
            "max_depth":         None,   # None = fully grown
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "max_features":      "sqrt",
            "random_state":      42,
        },
        "description":     "Ensemble of decision trees. Robust, low tuning effort.",
        "literature_name": "Random Forest",
    },

    "extra_trees": {
        "label":           "Extra Trees",
        "cls":             ExtraTreesRegressor,
        "group":           GROUP_ENSEMBLE,
        "default_params":  {
            "n_estimators":      100,
            "max_depth":         None,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "max_features":      "sqrt",
            "random_state":      42,
        },
        "description":     "Extremely Randomised Trees. Faster than RF, similar accuracy.",
        "literature_name": "Extra Trees",
    },

    "bagging": {
        "label":           "Bagging Regressor",
        "cls":             BaggingRegressor,
        "group":           GROUP_ENSEMBLE,
        "default_params":  {
            "n_estimators": 10,
            "max_samples":  1.0,
            "max_features": 1.0,
            "random_state": 42,
        },
        "description":     "Bootstrap aggregation over a base estimator.",
        "literature_name": "Bagging",
    },

    # ══════════════════════════════════════════════════════
    # Boosting family
    # Literature frequency: XGBoost 12x, Gradient Boosting 8x,
    #                       AdaBoost 2x, LightGBM 2x, CatBoost 3x
    # FIX: expanded xgb and gradient_boosting default_params.
    # ══════════════════════════════════════════════════════
    "xgb": {
        "label":           "Gradient Boosting (XGB-style)",
        "cls":             HistGradientBoostingRegressor,
        "group":           GROUP_BOOST,
        "default_params":  {
            "learning_rate":     0.05,
            "max_iter":          300,
            "max_depth":         None,   # None = no limit
            "min_samples_leaf":  20,
            "l2_regularization": 0.0,
            "random_state":      42,
        },
        "description":     "sklearn HistGradientBoosting — XGBoost-compatible, no extra dep.",
        "literature_name": "XGBoost",
    },

    "gradient_boosting": {
        "label":           "Gradient Boosting (classic)",
        "cls":             GradientBoostingRegressor,
        "group":           GROUP_BOOST,
        "default_params":  {
            "n_estimators":      100,
            "learning_rate":     0.1,
            "max_depth":         3,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "subsample":         1.0,
            "random_state":      42,
        },
        "description":     "sklearn GradientBoostingRegressor. Slower but more configurable.",
        "literature_name": "Gradient Boosting",
    },

    "adaboost": {
        "label":           "AdaBoost",
        "cls":             AdaBoostRegressor,
        "group":           GROUP_BOOST,
        "default_params":  {
            "n_estimators":  50,
            "learning_rate": 1.0,
            "random_state":  42,
        },
        "description":     "Adaptive Boosting. Good baseline for boosting family.",
        "literature_name": "Adaboost",
    },

    # ── LightGBM (optional install) ────────────────────────
    **({
        "lightgbm": {
            "label":           "LightGBM",
            "cls":             LGBMRegressor,
            "group":           GROUP_BOOST,
            "default_params":  {
                "n_estimators":     100,
                "learning_rate":    0.05,
                "max_depth":        -1,    # -1 = no limit in LightGBM
                "num_leaves":       31,
                "min_child_samples": 20,
                "reg_alpha":        0.0,
                "reg_lambda":       0.0,
                "random_state":     42,
                "verbose":          -1,
            },
            "description":     "Microsoft LightGBM. Fast gradient boosting on large datasets.",
            "literature_name": "Lightgbm",
        }
    } if _LIGHTGBM_OK else {}),

    # ── CatBoost (optional install) ────────────────────────
    **({
        "catboost": {
            "label":           "CatBoost",
            "cls":             CatBoostRegressor,
            "group":           GROUP_BOOST,
            "default_params":  {
                "iterations":    200,
                "learning_rate": 0.05,
                "depth":         6,
                "l2_leaf_reg":   3.0,
                "random_seed":   42,
                "verbose":       0,
            },
            "description":     "Yandex CatBoost. Handles categoricals natively.",
            "literature_name": "Catboost",
        }
    } if _CATBOOST_OK else {}),

    # ══════════════════════════════════════════════════════
    # Linear models
    # Literature frequency: Linear Regression 8x,
    #                       Polynomial Regression 1x,
    #                       Kernel Ridge 1x, Bayesian Ridge 1x
    # ══════════════════════════════════════════════════════
    "linear_reg": {
        "label":           "Linear Regression",
        "cls":             LinearRegression,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "fit_intercept": True,
        },
        "description":     "Ordinary least squares linear regression.",
        "literature_name": "Linear Regression",
    },

    "ridge": {
        "label":           "Ridge Regression",
        "cls":             Ridge,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "alpha": 1.0,
        },
        "description":     "L2-regularised linear regression. Stable baseline.",
        "literature_name": "Linear Regression",
    },

    "elasticnet": {
        "label":           "ElasticNet",
        "cls":             ElasticNet,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "alpha":    1.0,
            "l1_ratio": 0.5,
            "max_iter": 2000,
        },
        "description":     "L1+L2 combined regularisation. Good for sparse features.",
        "literature_name": "Linear Regression",
    },

    "poly_reg": {
        "label":           "Polynomial Regression",
        "cls":             PolynomialRegression,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "degree":        2,
            "fit_intercept": True,
        },
        "description":     "Linear regression on polynomial feature expansion (degree 2+).",
        "literature_name": "Polynomial Regression",
    },

    "bayesian_ridge": {
        "label":           "Bayesian Ridge",
        "cls":             BayesianRidge,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "alpha_1":  1e-6,
            "alpha_2":  1e-6,
            "lambda_1": 1e-6,
            "lambda_2": 1e-6,
        },
        "description":     "Bayesian linear regression with automatic regularisation.",
        "literature_name": "Bayesian Regression",
    },

    "kernel_ridge": {
        "label":           "Kernel Ridge Regression",
        "cls":             KernelRidge,
        "group":           GROUP_LINEAR,
        "default_params":  {
            "alpha":  1.0,
            "kernel": "rbf",
            "gamma":  None,   # None = 1/n_features
        },
        "description":     "Ridge regression in kernel space. Dual of kernel SVR.",
        "literature_name": "Kernel Ridge",
    },

    # ══════════════════════════════════════════════════════
    # Neural Network
    # Literature frequency: ANN 15x
    # FIX: expanded default_params to expose alpha and
    # learning_rate_init as manual controls.
    # ══════════════════════════════════════════════════════
    "mlp": {
        "label":           "MLP Neural Network (ANN)",
        "cls":             MLPRegressor,
        "group":           GROUP_NEURAL,
        "default_params":  {
            "hidden_layer_sizes": (100,),
            "alpha":              0.0001,
            "learning_rate_init": 0.001,
            "max_iter":           2000,
            "solver":             "adam",
            "random_state":       42,
        },
        "description":     "Shallow feed-forward neural network (ANN). sklearn MLPRegressor.",
        "literature_name": "ANN",
    },

    # ══════════════════════════════════════════════════════
    # Probabilistic
    # Literature frequency: Gaussian Process 2x
    # FIX: exposed alpha and n_restarts_optimizer.
    # ══════════════════════════════════════════════════════
    "gaussian_process": {
        "label":           "Gaussian Process Regression",
        "cls":             GaussianProcessRegressor,
        "group":           GROUP_PROBABILISTIC,
        "default_params":  {
            "alpha":                1e-10,  # noise level added to diagonal
            "n_restarts_optimizer": 0,
            "random_state":         42,
        },
        "description":     "GPR — gives prediction + uncertainty estimate. Slow on n>1000.",
        "literature_name": "Gaussian Process",
    },

    # ══════════════════════════════════════════════════════
    # Other
    # Literature frequency: KNN 9x
    # FIX: expanded default_params to expose weights and p.
    # ══════════════════════════════════════════════════════
    "knn": {
        "label":           "K-Nearest Neighbours",
        "cls":             KNeighborsRegressor,
        "group":           GROUP_OTHER,
        "default_params":  {
            "n_neighbors": 5,
            "weights":     "uniform",   # "uniform" | "distance"
            "p":           2,           # 1 = Manhattan, 2 = Euclidean
        },
        "description":     "Instance-based learner. Simple, no training phase.",
        "literature_name": "Knn",
    },
}


# ══════════════════════════════════════════════════════════
# Convenience helpers
# ══════════════════════════════════════════════════════════

def get_model_names_by_group() -> dict[str, list[str]]:
    """Return {group_label: [model_key, ...]} for grouped UI pickers."""
    groups: dict[str, list[str]] = {}
    for key, meta in MODEL_REGISTRY.items():
        groups.setdefault(meta["group"], []).append(key)
    return groups


def get_display_labels() -> dict[str, str]:
    """Return {model_key: label} for selectbox display."""
    return {k: v["label"] for k, v in MODEL_REGISTRY.items()}


def build_model(model_key: str, custom_params: dict | None = None):
    """
    Instantiate a model from the registry.

    Parameters
    ----------
    model_key    : key from MODEL_REGISTRY
    custom_params: override default_params (e.g. from autotune result)

    Returns
    -------
    Unfitted model instance ready for .fit()
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_key}' not found in registry. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    meta   = MODEL_REGISTRY[model_key]
    params = {**meta["default_params"], **(custom_params or {})}
    return meta["cls"](**params)


def models_for_paper(literature_algos: list[str]) -> list[str]:
    """
    Given a list of algorithm names as they appear in papers
    (e.g. ['ANN', 'Random Forest', 'SVM']), return matching
    registry keys.

    Used on the General Testing page when a paper's dataset
    is uploaded and the app auto-selects the relevant models.

    Example
    -------
    >>> models_for_paper(['ANN', 'Random Forest', 'XGBoost'])
    ['mlp', 'rf', 'xgb']
    """
    lit_name_to_keys: dict[str, list[str]] = {}
    for key, meta in MODEL_REGISTRY.items():
        lit = meta.get("literature_name", "").lower()
        lit_name_to_keys.setdefault(lit, []).append(key)

    matched = []
    for algo in literature_algos:
        keys = lit_name_to_keys.get(algo.strip().lower(), [])
        matched.extend(keys)

    # Deduplicate preserving order
    seen = set()
    return [k for k in matched if not (k in seen or seen.add(k))]


# ── Convenience key lists ──────────────────────────────────
ALL_MODEL_KEYS: list[str] = list(MODEL_REGISTRY.keys())

SVR_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_SVR
]

TREE_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_TREE
]

ENSEMBLE_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_ENSEMBLE
]

BOOST_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_BOOST
]

LINEAR_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_LINEAR
]

NEURAL_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_NEURAL
]

PROBABILISTIC_KEYS: list[str] = [
    k for k, v in MODEL_REGISTRY.items() if v["group"] == GROUP_PROBABILISTIC
]
