from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    r2_score,
    explained_variance_score,
    brier_score_loss,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
)


def evaluate_model(model, X_test, y_test, model_name, output_dir, labels=None, oof_predictions=None):
    """
    Evaluates classification models with comprehensive scoring mechanisms including:
    - Standard classification metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1)
    - Agreement and correlation coefficients (Cohen's Kappa, Matthews Correlation Coefficient)
    - Goodness-of-fit & calibration metrics (R2 Score, Explained Variance, Brier Score, Log Loss)
    - Statistical Z-Score significance tests (Z-score vs baseline, standardized residual Z-score)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if oof_predictions is not None:
        predictions = oof_predictions
    else:
        predictions = model.predict(X_test)

    if labels is None:
        labels = sorted(set(y_test) | set(predictions))

    n_samples = len(y_test)
    y_test_arr = np.array(y_test)
    predictions_arr = np.array(predictions)

    metrics = {
        "accuracy": float(accuracy_score(y_test_arr, predictions_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_arr, predictions_arr)),
        "macro_precision": float(precision_score(y_test_arr, predictions_arr, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_test_arr, predictions_arr, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_test_arr, predictions_arr, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_test_arr, predictions_arr, average="weighted", labels=labels, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_test_arr, predictions_arr)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test_arr, predictions_arr)),
        "classification_report": classification_report(y_test_arr, predictions_arr, labels=labels, output_dict=True, zero_division=0),
    }

    is_binary = len(labels) == 2
    if is_binary and hasattr(model, "predict_proba"):
        try:
            pos_label = labels[-1]
            proba = model.predict_proba(X_test)[:, -1]

            # Probabilistic Discrimination
            metrics["roc_auc"] = float(roc_auc_score(y_test_arr, proba))
            metrics["pr_auc"] = float(average_precision_score(y_test_arr, proba, pos_label=pos_label))

            # Scikit-learn Goodness-of-Fit & Calibration: R2 and Explained Variance
            metrics["r2_score"] = float(r2_score(y_test_arr, proba))
            metrics["explained_variance"] = float(explained_variance_score(y_test_arr, proba))

            # Brier Score Loss & Log Loss
            brier = float(brier_score_loss(y_test_arr, proba))
            metrics["brier_score"] = brier
            metrics["log_loss"] = float(log_loss(y_test_arr, proba))

            # Brier Skill Score (BSS) relative to climatological base rate
            p_base = float(np.mean(y_test_arr == pos_label))
            bs_ref = p_base * (1.0 - p_base)
            metrics["brier_skill_score"] = float(1.0 - (brier / bs_ref)) if bs_ref > 0 else 0.0

            # Statistical Z-Score Significance (Accuracy vs Majority Class Baseline)
            acc = metrics["accuracy"]
            p0 = max(p_base, 1.0 - p_base)  # majority baseline proportion
            se_base = np.sqrt(p0 * (1.0 - p0) / n_samples)
            metrics["z_score_accuracy"] = float((acc - p0) / se_base) if se_base > 0 else 0.0

            # Standardized Residual Z-Score (testing systematic probability over/under forecasting)
            residuals = y_test_arr - proba
            std_res = np.std(residuals)
            se_res = std_res / np.sqrt(n_samples)
            metrics["z_score_residual"] = float(np.mean(residuals) / se_res) if se_res > 0 else 0.0
        except Exception as e:
            print(f"Warning computing probability metrics for {model_name}: {e}")

    cm = confusion_matrix(y_test_arr, predictions_arr, labels=labels)
    label_names = ["No Landslide", "Landslide"] if is_binary else ["Low", "Moderate", "High"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close()

    print(f"Confusion matrix saved for {model_name}")

    return metrics
