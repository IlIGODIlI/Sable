# Training Summary

## Datasets
- Integrated `landslide_dataset.csv` with existing datasets, resulting in a combined dataset of **464,148** samples and **29** features.

## Models Trained
| Model | ROC‑AUC | R² | Z‑Score | Brier Score | Accuracy |
|-------|---------|-----|---------|-------------|----------|
| Logistic Regression | 0.7789 | 0.1587 | 27.94 | 0.1918 | 73.2% |
| Random Forest | 0.9755 | 0.7197 | 169.33 | 0.0639 | 91.38% |
| XGBoost | 0.9703 | 0.6920 | 161.62 | 0.0702 | 90.18% |

## Dynamic Hyper‑Parameter Knowledge Base
A dictionary of hyper‑parameters and their meanings has been saved to `outputs/model_parameters.json`. Key examples:
- `C` (Logistic Regression): Inverse of regularization strength; higher values reduce regularization.
- `n_estimators` (Random Forest & XGBoost): Number of trees in the ensemble.
- `max_depth`: Maximum depth of each tree; controls model complexity.
- `learning_rate` (XGBoost): Step size shrinkage used in gradient boosting.

## Evaluation Artifacts
- Confusion‑matrix images for each model are stored under `outputs/`.
- Full metric collection (`ROC‑AUC`, `R²`, `explained variance`, `Brier score`, `Z‑score`) is recorded in `outputs/results.json`.

## Next Steps
- Review the generated `training_summary.md` for any additional details you need.
- If you require model export to another format or a batch inference script, let me know.
