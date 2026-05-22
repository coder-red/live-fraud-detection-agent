"""Train the fraud model and generate reproducible model artifacts.

This script mirrors the notebook training flow while producing stable outputs
for model serving, charts, and evaluation reports.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, MODEL_PARAMS, RAW_DATA_PATH
from src.features import preprocess_features


ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_THRESHOLD = 0.5221
TRAIN_ROWS = 500_000


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH, nrows=TRAIN_ROWS)


def train_model(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_raw, test_raw = train_test_split(
        df,
        test_size=0.2,
        stratify=df["is_fraud"],
        random_state=42,
    )

    X_train = preprocess_features(train_raw)
    y_train = train_raw["is_fraud"]
    X_test = preprocess_features(test_raw)
    y_test = test_raw["is_fraud"]

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    return model, X_train, y_train, X_test, y_test


def optimal_threshold(probs: np.ndarray, y_test: pd.Series) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def save_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        cbar=False,
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
    )
    plt.title("Confusion Matrix: Fraud Detection Performance", fontsize=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion.png", dpi=150)
    plt.close()


def save_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    feat_importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 8))
    feat_importance.head(15).plot(kind="barh", x="feature", y="importance")
    plt.gca().invert_yaxis()
    plt.title("Top 15 Features (Gain)")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "features.png", dpi=150)
    plt.close()

    feat_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    return feat_importance


def save_reports(
    probs: np.ndarray,
    y_test: pd.Series,
    preds: np.ndarray,
    best_threshold: float,
    best_f1: float,
) -> None:
    report = classification_report(y_test, preds, digits=4, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    metrics = {
        "data_source": str(RAW_DATA_PATH),
        "train_rows": TRAIN_ROWS,
        "serving_threshold": DEFAULT_THRESHOLD,
        "best_f1_threshold": best_threshold,
        "best_f1_score": best_f1,
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "model_params": MODEL_PARAMS,
    }

    (REPORTS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(cm)


def save_model_artifacts(model: xgb.XGBClassifier, feature_names: list[str]) -> None:
    model.save_model(MODEL_DIR / "fraud_model.json")
    joblib.dump(feature_names, MODEL_DIR / "feature_list.pkl")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_data()
    model, X_train, _y_train, X_test, y_test = train_model(df)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= DEFAULT_THRESHOLD).astype(int)
    best_threshold, best_f1 = optimal_threshold(probs, y_test)

    save_model_artifacts(model, list(X_train.columns))
    save_feature_importance(model, list(X_train.columns))
    save_reports(probs, y_test, preds, best_threshold, best_f1)

    print(f"Saved model artifacts to {MODEL_DIR}")
    print(f"Saved charts to {ASSETS_DIR}")
    print(f"Saved reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
