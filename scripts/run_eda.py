"""Generate EDA charts and summary outputs from the fraud training dataset.

This script replaces the exploratory notebook workflow with a reproducible,
versioned command that writes chart artifacts into `assets/` and summary data
into `reports/`.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, RAW_DATA_PATH


ASSETS_DIR = BASE_DIR / "assets"


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_name"] = df["trans_date_trans_time"].dt.day_name()
    return df


def save_target_distribution(df: pd.DataFrame) -> dict:
    fraud_counts = df["is_fraud"].value_counts().sort_index()
    fraud_pct = (df["is_fraud"].value_counts(normalize=True).sort_index() * 100).round(4)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=fraud_pct.index, y=fraud_pct.values)
    plt.title("Target Distribution (Imbalance Check)")
    plt.xlabel("is_fraud")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "target.png", dpi=150)
    plt.close()

    return {
        "total_transactions": int(len(df)),
        "legitimate_count": int(fraud_counts.get(0, 0)),
        "fraud_count": int(fraud_counts.get(1, 0)),
        "legitimate_pct": float(fraud_pct.get(0, 0.0)),
        "fraud_pct": float(fraud_pct.get(1, 0.0)),
    }


def save_hourly_fraud_chart(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df[df["is_fraud"] == 0], x="hour", label="Legit", fill=True, clip=(0, 23))
    sns.kdeplot(data=df[df["is_fraud"] == 1], x="hour", label="Fraud", fill=True, clip=(0, 23))
    plt.title("Fraudulent Activity by Hour of Day")
    plt.xticks(range(0, 24))
    plt.xlabel("Hour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fraud_by_hour.png", dpi=150)
    plt.close()


def save_category_fraud_chart(df: pd.DataFrame) -> list[dict]:
    cat_fraud = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    cat_fraud.plot(kind="bar", color="salmon")
    plt.axhline(df["is_fraud"].mean(), color="blue", linestyle="--", label="Global Avg")
    plt.title("Fraud Probability by Category")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fraud_by_category.png", dpi=150)
    plt.close()

    return [
        {"category": category, "fraud_rate": float(rate)}
        for category, rate in cat_fraud.head(10).items()
    ]


def save_density_heatmap(df: pd.DataFrame) -> None:
    fraud_only = df[df["is_fraud"] == 1]
    heatmap_data = fraud_only.groupby(["day_name", "hour"]).size().unstack(fill_value=0)
    fraud_total = heatmap_data.sum().sum()
    fraud_pct = (heatmap_data / fraud_total * 100).round(1) if fraud_total else heatmap_data

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=fraud_pct,
        fmt=".1f",
        cbar_kws={"label": "Fraud Count"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Fraud Density: Day of Week vs. Hour of Day", fontsize=15, fontweight="bold")
    plt.xlabel("Hour of Day (24h Format)", fontsize=12)
    plt.ylabel("Day of Week", fontsize=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fraud_density_heatmap.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    pd.set_option("display.max_columns", None)

    df = load_data()
    save_target_distribution(df)
    save_hourly_fraud_chart(df)
    save_category_fraud_chart(df)
    save_density_heatmap(df)

    print(f"Wrote EDA assets to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
