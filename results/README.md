# Results

This folder stores all experiment outputs after training is complete.

## Files added after experiments

### Metrics
| File | Description |
|---|---|
| baseline_metrics.csv | Locked baseline XGBoost scores across 5 folds |
| e_xgboost_metrics.csv | E-XGBoost scores across 5 folds |
| comparison_table.csv | Full comparison with delta and p-values |

### Plots
| File | Description |
|---|---|
| roc_curves.png | ROC curves for all models |
| confusion_matrix_baseline.png | Normalised confusion matrix — baseline |
| confusion_matrix_e_xgboost.png | Normalised confusion matrix — E-XGBoost |
| shap_summary.png | SHAP global feature importance plot |
| shap_waterfall.png | SHAP local per-prediction explanation |
| learning_curves.png | Training vs validation loss curves |

## Expected results table
| Metric | Baseline XGBoost | E-XGBoost | Delta | p-value | Significant |
|---|---|---|---|---|---|
| Accuracy | TBD | TBD | TBD | TBD | TBD |
| Macro F1 | TBD | TBD | TBD | TBD | TBD |
| AUC-ROC | TBD | TBD | TBD | TBD | TBD |
| AUC-PR | TBD | TBD | TBD | TBD | TBD |
| Training Time | TBD | TBD | TBD | — | — |

Note: All metrics reported as mean ± std over 5-fold stratified
cross-validation. Statistical significance tested using Wilcoxon
Signed-Rank Test (p < 0.05).

## Reproducibility note
All results generated using:
- SEED = 42
- Python 3.9
- xgboost 1.7.0
- Kaggle Notebooks free tier
- Identical train/val/test splits for baseline and E-XGBoost
