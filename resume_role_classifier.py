"""Train and evaluate a resume job-category classifier.

This educational project predicts the dataset's resume category. It does not
score a candidate against a job description or make a hiring recommendation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "results" / ".matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Resume.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RANDOM_STATE = 42


def clean_resume(text: str) -> str:
    """Remove common contact details and normalize whitespace."""
    text = str(text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAIL ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\+?\d[\d().\s-]{7,}\d", " PHONE ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_model() -> Pipeline:
    """Create the text features and the project's single ML classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=15000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
            ),
        ]
    )


def plot_category_counts(counts: pd.Series) -> None:
    ordered = counts.sort_values()
    height = max(7.0, len(ordered) * 0.28)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(ordered.index, ordered.values, color="#167D8D")
    ax.set_title("Resume examples by category", loc="left", fontsize=16, pad=14)
    ax.set_xlabel("Number of resumes")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "category_distribution.png", dpi=180)
    plt.close(fig)


def plot_category_f1(labels: list[str], f1_values: np.ndarray) -> None:
    order = np.argsort(f1_values)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(np.array(labels)[order], f1_values[order], color="#E08E45")
    ax.set_xlim(0, 1)
    ax.set_title("Held-out F1 by category", loc="left", fontsize=16, pad=14)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "category_f1.png", dpi=180)
    plt.close(fig)


def train_and_evaluate(data_path: Path = DATA_PATH) -> tuple[Pipeline, dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download Resume.csv into the data folder."
        )

    data = pd.read_csv(data_path)
    required = {"Resume_str", "Category"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data[["Resume_str", "Category"]].dropna().copy()
    data["Resume_str"] = data["Resume_str"].map(clean_resume)
    data["Category"] = data["Category"].astype(str).str.strip()
    data = data[(data["Resume_str"] != "") & (data["Category"] != "")]
    counts = data["Category"].value_counts().sort_index()

    x_train, x_test, y_train, y_test = train_test_split(
        data["Resume_str"],
        data["Category"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["Category"],
    )

    model = make_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    labels = sorted(data["Category"].unique())
    accuracy = float(np.mean(np.asarray(predictions) == y_test.to_numpy()))
    macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_test, predictions, labels=labels, zero_division=0
    )

    per_category = pd.DataFrame(
        {
            "category": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1_values,
            "test_examples": support,
        }
    )
    per_category.to_csv(RESULTS_DIR / "per_category_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv", index_label="actual_category")

    metrics = {
        "dataset_rows_used": int(len(data)),
        "number_of_categories": int(data["Category"].nunique()),
        "training_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "algorithm": "Logistic Regression",
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0),
        encoding="utf-8",
    )
    counts.to_csv(RESULTS_DIR / "category_counts.csv", header=["resume_count"])
    plot_category_counts(counts)
    plot_category_f1(labels, f1_values)

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-text",
        help="Optional resume text to classify after training, in quotation marks.",
    )
    args = parser.parse_args()

    model, metrics = train_and_evaluate()
    print("Resume category classifier evaluation")
    print(f"Examples: {metrics['dataset_rows_used']}")
    print(f"Categories: {metrics['number_of_categories']}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Results saved in: {RESULTS_DIR}")

    if args.demo_text:
        predicted = model.predict([clean_resume(args.demo_text)])[0]
        probabilities = model.predict_proba([clean_resume(args.demo_text)])[0]
        best_score = float(np.max(probabilities))
        print(f"\nPredicted category: {predicted}")
        print(f"Highest class probability: {best_score:.3f}")
        print("This estimate may not be calibrated and is not a measure of job suitability.")


if __name__ == "__main__":
    main()
