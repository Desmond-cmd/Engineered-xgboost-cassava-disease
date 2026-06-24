# Notebooks

Run these notebooks in strict order. Do NOT skip steps.

## Order of execution

### 1. 01_baseline_xgboost.ipynb
- Builds the standard XGBoost model with default settings
- NO Focal Loss — this is the scientific control
- Run once, save metrics, NEVER modify again after locking
- Saves results to `results/baseline_metrics.csv`

### 2. 02_e_xgboost_focal_loss.ipynb
- Builds E-XGBoost with custom Focal Loss objective
- Uses IDENTICAL data splits and SEED=42 as notebook 01
- Imports focal_loss_objective from `src/focal_loss.py`
- Saves results to `results/e_xgboost_metrics.csv`

### 3. 03_shap_explainability.ipynb
- Loads trained E-XGBoost model
- Generates SHAP summary plot (global feature importance)
- Generates SHAP waterfall plot (per-prediction explanation)
- Saves plots to `results/`

## Critical rules
- SEED = 42 in every notebook — never change this
- Baseline notebook is LOCKED after first run
- All notebooks use identical train/val/test splits
- 5-fold stratified cross-validation throughout

## Computing environment
- Platform: Kaggle Notebooks (free tier)
- CPU: sufficient for XGBoost (no GPU needed)
- RAM: 16GB available on Kaggle
- Estimated runtime: ~30 minutes total
