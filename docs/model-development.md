# Model Development Workflow

This document describes the reproducible, script-based workflow for generating the model artifacts and analysis assets used by this repository.

## Goals

- Replace notebook-only provenance with versioned scripts.
- Keep `assets/` charts tied to concrete commands.
- Keep `model/` artifacts tied to a repeatable training flow.

## Commands

Run the EDA pipeline:

```bash
uv run python scripts/run_eda.py
```

Run the training pipeline:

```bash
uv run python scripts/train_model.py
```

## Generated Outputs

### EDA outputs

`scripts/run_eda.py` reads `data/fraudTrain.csv` and generates:

- `assets/target.png`
- `assets/fraud_by_hour.png`
- `assets/fraud_by_category.png`
- `assets/fraud_density_heatmap.png`

### Training outputs

`scripts/train_model.py` reads `data/fraudTrain.csv`, trains the XGBoost model on the first `500000` rows, evaluates it, and generates:

- `model/fraud_model.json`
- `model/feature_list.pkl`
- `assets/confusion.png`
- `assets/features.png`
- `reports/training_metrics.json`
- `reports/feature_importance.csv`

`reports/training_metrics.json` stores:

- ROC-AUC
- the serving threshold used by the app (`0.5221`)
- the best threshold found by F1 search on the evaluation split
- the confusion matrix
- the full classification report
- the model parameters used for training

`reports/feature_importance.csv` stores: the feature importance values, the feature names, and the feature types.
