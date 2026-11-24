"""
Update all stale hardcoded metric fallback values in app.py
to match the live values from outputs/results.json.
"""

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# ── 1. Update the fallback metrics dict (used when results.json missing) ───
OLD_FALLBACK = '''"logistic_regression": {
                "accuracy": 0.6998, "balanced_accuracy": 0.7042, "roc_auc": 0.7835,
                "r2_score": 0.1250, "z_score_accuracy": 31.5, "brier_score": 0.2014,
                "macro_f1": 0.6879, "matthews_corrcoef": 0.3912,
            },
            "random_forest": {
                "accuracy": 0.9144, "balanced_accuracy": 0.9192, "roc_auc": 0.9755,
                "r2_score": 0.6854, "z_score_accuracy": 169.8, "brier_score": 0.0712,
                "macro_f1": 0.9083, "matthews_corrcoef": 0.8175,
            },
            "xgboost": {
                "accuracy": 0.9014, "balanced_accuracy": 0.9079, "roc_auc": 0.9699,
                "r2_score": 0.6421, "z_score_accuracy": 158.4, "brier_score": 0.0815,
                "macro_f1": 0.8949, "matthews_corrcoef": 0.7915,
            },'''

NEW_FALLBACK = '''"logistic_regression": {
                "accuracy": 0.6923, "balanced_accuracy": 0.6988, "roc_auc": 0.7787,
                "r2_score": 0.1587, "z_score_accuracy": 27.9, "brier_score": 0.1918,
                "macro_f1": 0.6813, "matthews_corrcoef": 0.3808,
            },
            "random_forest": {
                "accuracy": 0.9138, "balanced_accuracy": 0.9183, "roc_auc": 0.9755,
                "r2_score": 0.7197, "z_score_accuracy": 169.3, "brier_score": 0.0639,
                "macro_f1": 0.9077, "matthews_corrcoef": 0.8186,
            },
            "xgboost": {
                "accuracy": 0.9018, "balanced_accuracy": 0.9080, "roc_auc": 0.9703,
                "r2_score": 0.6920, "z_score_accuracy": 161.6, "brier_score": 0.0702,
                "macro_f1": 0.8953, "matthews_corrcoef": 0.7953,
            },'''

content = content.replace(OLD_FALLBACK, NEW_FALLBACK, 1)

# ── 2. Update inline .get() fallback defaults in the HTML chip/table rows ──
# Logistic Regression
content = content.replace("lr_meta.get('accuracy', 0.6998)", "lr_meta.get('accuracy', 0.6923)")
content = content.replace("lr_meta.get('balanced_accuracy', 0.7040)", "lr_meta.get('balanced_accuracy', 0.6988)")
content = content.replace("lr_meta.get('balanced_accuracy', 0.7042)", "lr_meta.get('balanced_accuracy', 0.6988)")
content = content.replace("lr_meta.get('roc_auc', 0.7833)", "lr_meta.get('roc_auc', 0.7787)")
content = content.replace("lr_meta.get('roc_auc', 0.7835)", "lr_meta.get('roc_auc', 0.7787)")
content = content.replace("lr_meta.get('r2_score', 0.125)", "lr_meta.get('r2_score', 0.1587)")
content = content.replace("lr_meta.get('r2_score', 0.1250)", "lr_meta.get('r2_score', 0.1587)")
content = content.replace("lr_meta.get('z_score_accuracy', 31.5)", "lr_meta.get('z_score_accuracy', 27.9)")
content = content.replace("lr_meta.get('brier_score', 0.201)", "lr_meta.get('brier_score', 0.1918)")
content = content.replace("lr_meta.get('brier_score', 0.2014)", "lr_meta.get('brier_score', 0.1918)")

# Random Forest
content = content.replace("rf_meta.get('accuracy', 0.9144)", "rf_meta.get('accuracy', 0.9138)")
content = content.replace("rf_meta.get('balanced_accuracy', 0.9192)", "rf_meta.get('balanced_accuracy', 0.9183)")
content = content.replace("rf_meta.get('roc_auc', 0.9755)", "rf_meta.get('roc_auc', 0.9755)")
content = content.replace("rf_meta.get('r2_score', 0.6854)", "rf_meta.get('r2_score', 0.7197)")
content = content.replace("rf_meta.get('z_score_accuracy', 169.8)", "rf_meta.get('z_score_accuracy', 169.3)")
content = content.replace("rf_meta.get('brier_score', 0.0712)", "rf_meta.get('brier_score', 0.0639)")

# XGBoost
content = content.replace("xgb_meta.get('accuracy', 0.9014)", "xgb_meta.get('accuracy', 0.9018)")
content = content.replace("xgb_meta.get('balanced_accuracy', 0.9079)", "xgb_meta.get('balanced_accuracy', 0.9080)")
content = content.replace("xgb_meta.get('roc_auc', 0.9699)", "xgb_meta.get('roc_auc', 0.9703)")
content = content.replace("xgb_meta.get('r2_score', 0.6421)", "xgb_meta.get('r2_score', 0.6920)")
content = content.replace("xgb_meta.get('z_score_accuracy', 158.4)", "xgb_meta.get('z_score_accuracy', 161.6)")
content = content.replace("xgb_meta.get('brier_score', 0.0815)", "xgb_meta.get('brier_score', 0.0702)")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done.")
