from pathlib import Path
import json
from scipy.stats import loguniform, uniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Comprehensive knowledge base describing control parameters, their mathematical meanings,
# and their operational purposes in landslide susceptibility prediction.
PARAMETER_METADATA = {
    "logistic_regression": {
        "model_name": "Logistic Regression",
        "model_type": "Generalized Linear Classifier",
        "description": "Baseline linear statistical classifier with sigmoid link function and L2 regularization.",
        "parameters": {
            "C": {
                "display_name": "Inverse Regularization Strength (C)",
                "control_type": "Complexity Penalty",
                "type": "float",
                "meaning": "Specifies the inverse of regularization strength (1 / λ). Smaller values impose stronger L2 penalization, heavily shrinking regression coefficients toward zero.",
                "purpose": "Prevents overfitting on collinear geospatial and geomorphic factors (e.g., high correlation between elevation, slope, and terrain ruggedness) by dampening uninformative weights."
            },
            "solver": {
                "display_name": "Numerical Solver Algorithm",
                "control_type": "Loss Optimization",
                "type": "string",
                "meaning": "Optimization routine used to minimize cross-entropy loss (e.g., L-BFGS: Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton algorithm).",
                "purpose": "Provides rapid, numerically stable convergence on large multidimensional raster datasets containing continuous topographic indices."
            },
            "max_iter": {
                "display_name": "Maximum Iterations",
                "control_type": "Convergence Boundary",
                "type": "integer",
                "meaning": "Maximum number of optimization gradient steps allowed before terminating search.",
                "purpose": "Guarantees full mathematical convergence on dense feature surfaces without premature stopping."
            },
            "class_weight": {
                "display_name": "Class Weighting",
                "control_type": "Cost-Sensitive Balancing",
                "type": "string",
                "meaning": "Strategy to weight loss penalties inversely proportional to class frequencies across training samples.",
                "purpose": "Heavily penalizes false negatives in hazard zones when stable terrain samples substantially outnumber landslide events, elevating sensitivity for early warning."
            },
            "penalty": {
                "display_name": "Regularization Norm",
                "control_type": "Loss Regularization",
                "type": "string",
                "meaning": "Norm utilized in penalization (L2 Ridge penalty: sum of squared weight magnitudes).",
                "purpose": "Distributes feature weights smoothly across correlated terrain, radar, and vegetation bands rather than abruptly dropping features."
            }
        }
    },
    "random_forest": {
        "model_name": "Random Forest",
        "model_type": "Bagging Ensemble of Decision Trees",
        "description": "Ensemble of decorrelated decision trees trained via bootstrap aggregation (bagging) with random feature subspace sampling.",
        "parameters": {
            "n_estimators": {
                "display_name": "Number of Decision Trees (n_estimators)",
                "control_type": "Ensemble Capacity",
                "type": "integer",
                "meaning": "Total count of decorrelated decision trees constructed in the bagging ensemble.",
                "purpose": "Suppresses overall ensemble variance, producing robust, noise-resilient spatial susceptibility classifications across varied geographic terrain."
            },
            "max_depth": {
                "display_name": "Maximum Tree Depth (max_depth)",
                "control_type": "Tree Capacity",
                "type": "integer",
                "meaning": "Maximum vertical hierarchical branching limit from root to terminal leaf in any tree.",
                "purpose": "Constrains hierarchical partitioning depth, preventing individual trees from memorizing localized topographical micro-features and acquisition artifacts."
            },
            "min_samples_split": {
                "display_name": "Minimum Split Samples (min_samples_split)",
                "control_type": "Structural Regularization",
                "type": "integer",
                "meaning": "Minimum number of observation samples required to consider splitting an internal tree node.",
                "purpose": "Prevents spurious bifurcations driven by localized geospatial outliers or radar speckle in Sentinel imagery."
            },
            "min_samples_leaf": {
                "display_name": "Minimum Leaf Samples (min_samples_leaf)",
                "control_type": "Leaf Smoothing",
                "type": "integer",
                "meaning": "Minimum number of observation samples that must reside in any terminal leaf node.",
                "purpose": "Enforces spatial consensus smoothing, ensuring hazard predictions are supported by statistically significant contiguous clusters of terrain pixels."
            },
            "max_features": {
                "display_name": "Feature Subspace Ratio (max_features)",
                "control_type": "Feature Decorrelation",
                "type": "string_or_float",
                "meaning": "Number or fraction of feature dimensions randomly evaluated when seeking the optimal partition at each split.",
                "purpose": "Ensures diverse geomorphic, spectral (NDVI/EVI), and radar texture (GLCM) features are explored, preventing slope gradient alone from monopolizing splits."
            },
            "criterion": {
                "display_name": "Splitting Impurity Criterion",
                "control_type": "Partition Quality",
                "type": "string",
                "meaning": "Mathematical metric (e.g. Gini Impurity) used to evaluate the information purity gained by a potential split.",
                "purpose": "Maximizes the separation between stable terrain and high-risk landslide scars at each stage of tree construction."
            },
            "class_weight": {
                "display_name": "Class Weighting Scheme",
                "control_type": "Cost-Sensitive Balancing",
                "type": "string",
                "meaning": "Adjusts loss weights inversely proportional to target class distribution across trees.",
                "purpose": "Prioritizes correct identification of sparse landslide events, reducing dangerous false-negative risk assessments."
            }
        }
    },
    "xgboost": {
        "model_name": "XGBoost Classifier",
        "model_type": "Gradient Boosted Decision Trees",
        "description": "Sequential boosting algorithm that trains shallow trees to minimize second-order gradient loss residuals with explicit regularization.",
        "parameters": {
            "n_estimators": {
                "display_name": "Boosting Iterations (n_estimators)",
                "control_type": "Ensemble Capacity",
                "type": "integer",
                "meaning": "Number of sequential gradient boosted decision trees fitted to residual errors of previous iterations.",
                "purpose": "Successively refines subtle non-linear boundaries separating unstable slope regimes from surrounding stable geomorphology."
            },
            "learning_rate": {
                "display_name": "Shrinkage Rate (η / learning_rate)",
                "control_type": "Optimization Step Size",
                "type": "float",
                "meaning": "Scaling factor applied to the leaf weights of each newly constructed tree before updating model state.",
                "purpose": "Prevents aggressive overshooting of optimal loss parameters, enabling gradual generalization across diverse multi-state topographies."
            },
            "max_depth": {
                "display_name": "Tree Maximum Depth (max_depth)",
                "control_type": "Interaction Order",
                "type": "integer",
                "meaning": "Maximum tree depth, governing the highest order of feature interactions modeled by any individual weak learner.",
                "purpose": "Constrains complex non-linear interactions (e.g., slope × curvature × TWI) to realistic geomorphic limits, preventing overfitting."
            },
            "subsample": {
                "display_name": "Instance Subsampling (subsample)",
                "control_type": "Stochastic Regularization",
                "type": "float",
                "meaning": "Fraction of training samples randomly drawn without replacement to train each sequential boosting round.",
                "purpose": "Introduces stochastic perturbation to desensitize the model against atmospheric distortion and Sentinel sensor noise."
            },
            "colsample_bytree": {
                "display_name": "Feature Subsampling (colsample_bytree)",
                "control_type": "Feature Regularization",
                "type": "float",
                "meaning": "Subsample ratio of feature columns evaluated when constructing each boosting tree.",
                "purpose": "Prevents high-contrast SAR backscatter features from dominating subtle vegetation and moisture index signatures."
            },
            "gamma": {
                "display_name": "Minimum Split Loss Reduction (γ)",
                "control_type": "Tree Pruning",
                "type": "float",
                "meaning": "Minimum loss reduction required to make a further partition on an existing leaf node.",
                "purpose": "Prunes marginal tree branches that do not yield significant information gain, keeping the hazard model parsimonious."
            },
            "min_child_weight": {
                "display_name": "Minimum Child Weight (min_child_weight)",
                "control_type": "Hessian Regularization",
                "type": "integer",
                "meaning": "Minimum sum of instance second derivatives (hessian weights) required in a child leaf node.",
                "purpose": "Prevents boosting trees from carving out isolated leaves for small localized DEM anomalies."
            },
            "scale_pos_weight": {
                "display_name": "Positive Class Multiplier (scale_pos_weight)",
                "control_type": "Cost-Sensitive Balancing",
                "type": "float",
                "meaning": "Multiplier applied to gradient weights for positive class observations (landslide events).",
                "purpose": "Calibrates the model's sensitivity threshold, heavily penalizing missed landslides to ensure reliable hazard alerts."
            },
            "objective": {
                "display_name": "Loss Objective",
                "control_type": "Mathematical Formulation",
                "type": "string",
                "meaning": "Loss function to optimize (binary:logistic outputs calibrated class probabilities).",
                "purpose": "Produces calibrated posterior risk probabilities (0.0 to 1.0) directly interpretable for emergency warning maps."
            }
        }
    }
}


def get_dynamic_param_distributions(scale_pos_weight=1.85):
    """
    Returns dynamic parameter distributions for scikit-learn's RandomizedSearchCV.
    Uses continuous (loguniform, uniform) and discrete (randint, lists) distributions
    to dynamically explore the hyperparameter space.
    """
    return {
        "logistic_regression": {
            "model__C": loguniform(1e-2, 1e2),
            "model__max_iter": [500, 1000, 1500],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": randint(150, 350),
            "model__max_depth": randint(8, 20),
            "model__min_samples_split": randint(2, 12),
            "model__min_samples_leaf": randint(4, 16),
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
        },
        "xgboost": {
            "model__n_estimators": randint(200, 450),
            "model__max_depth": randint(4, 9),
            "model__learning_rate": uniform(0.02, 0.12),
            "model__subsample": uniform(0.65, 0.30),
            "model__colsample_bytree": uniform(0.65, 0.30),
            "model__gamma": uniform(0.0, 0.4),
            "model__min_child_weight": randint(1, 8),
            "model__scale_pos_weight": [scale_pos_weight],
        },
    }


def clean_param_key(key):
    """Strips pipeline prefix 'model__' if present."""
    if key.startswith("model__"):
        return key[len("model__"):]
    return key


def format_param_value(val):
    """Formats parameter value for clean JSON and UI display."""
    if isinstance(val, (float,)):
        return round(float(val), 4)
    if isinstance(val, (int, bool, str)):
        return val
    return str(val)


def build_parameter_dictionary(model_name, best_params, pipeline=None):
    """
    Constructs a rich dictionary for a model, mapping each identified parameter
    to its tuned value, data type, functional control type, mathematical meaning,
    and operational landslide susceptibility prediction purpose.
    """
    meta = PARAMETER_METADATA.get(model_name, {})
    param_meta = meta.get("parameters", {})

    extracted_params = {}
    # First, record best parameters found during search
    for k, v in best_params.items():
        clean_k = clean_param_key(k)
        extracted_params[clean_k] = v

    # If pipeline provided, extract any relevant static/fitted parameters that complement the search
    if pipeline is not None and "model" in pipeline.named_steps:
        estimator = pipeline.named_steps["model"]
        est_params = estimator.get_params()
        for k in param_meta.keys():
            if k not in extracted_params and k in est_params:
                extracted_params[k] = est_params[k]

    result_params = {}
    for param_name, param_val in extracted_params.items():
        param_info = param_meta.get(param_name, {
            "display_name": param_name,
            "control_type": "Model Hyperparameter",
            "type": type(param_val).__name__,
            "meaning": f"Controls the {param_name} behavior of {model_name}.",
            "purpose": f"Regulates {param_name} during landslide susceptibility inference."
        })

        result_params[param_name] = {
            "value": format_param_value(param_val),
            "display_name": param_info.get("display_name", param_name),
            "control_type": param_info.get("control_type", "Hyperparameter"),
            "type": param_info.get("type", type(param_val).__name__),
            "meaning": param_info.get("meaning", ""),
            "purpose": param_info.get("purpose", "")
        }

    return {
        "model_key": model_name,
        "model_name": meta.get("model_name", model_name),
        "model_type": meta.get("model_type", "Machine Learning Estimator"),
        "description": meta.get("description", ""),
        "parameters": result_params
    }


def save_parameter_dictionary(param_dict, output_path):
    """Saves the parameter dictionary to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(param_dict, f, indent=2)


def load_parameter_dictionary(path=None):
    """
    Loads the parameter dictionary from JSON file, falling back to default
    metadata if file does not exist.
    """
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fallback to default dictionary populated with standard baseline values
    defaults = {
        "logistic_regression": {
            "C": 1.0, "max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced", "penalty": "l2"
        },
        "random_forest": {
            "n_estimators": 300, "max_depth": 15, "min_samples_leaf": 10,
            "min_samples_split": 2, "max_features": "sqrt", "class_weight": "balanced", "criterion": "gini"
        },
        "xgboost": {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.85, "objective": "binary:logistic"
        }
    }

    full_dict = {}
    for model_name, p_vals in defaults.items():
        full_dict[model_name] = build_parameter_dictionary(model_name, p_vals)
    return full_dict
