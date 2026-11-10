"""Reproduce the leakage-free Iris KNN homework experiment (Parts 3-5).

Run ``python knn_analysis.py`` from any working directory. All five reports are
written beside this script in ``reports/``. The invented Part 2 example and the
hypothetical Part 6 matrix are deliberately separate from this measured study.
"""

from __future__ import annotations

import csv
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
K_VALUES = (1, 3, 5, 7, 9, 15, 25)
LABELS = (0, 1, 2)
SPECIES = ("setosa", "versicolor", "virginica")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write an exact-column CSV, preserving the supplied row order."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pipeline_for(k: int) -> Pipeline:
    """Fit scaling and Euclidean, uniform-vote KNN as one pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                    algorithm="brute",
                ),
            ),
        ]
    )


def print_partition(name: str, indices: np.ndarray, y: np.ndarray) -> None:
    counts = np.bincount(y[indices], minlength=len(LABELS))
    details = ", ".join(f"{species}={count}" for species, count in zip(SPECIES, counts))
    print(f"  {name:>10}: {len(indices):3d} rows | {details}")


def save_selection_chart(rows: list[dict]) -> None:
    ks = [row["k"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [row["train_accuracy"] for row in rows], "o-", label="Training")
    ax.plot(
        ks,
        [row["validation_accuracy"] for row in rows],
        "s-",
        label="Validation",
    )
    ax.set(
        title="Iris KNN: choose k using validation data",
        xlabel="Number of neighbors (k)",
        ylabel="Accuracy",
        xticks=ks,
        ylim=(0.75, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "k_selection.png", dpi=180)
    plt.close(fig)


def save_confusion_charts(matrices: list[tuple[str, np.ndarray]], best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    max_count = max(int(matrix.max()) for _, matrix in matrices)
    for ax, (model_name, matrix) in zip(axes, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        title = f"KNN (k={best_k})" if model_name == "knn" else "Most-frequent baseline"
        ax.set(
            title=title,
            xlabel="Predicted species",
            ylabel="Actual species",
            xticks=range(len(SPECIES)),
            yticks=range(len(SPECIES)),
            xticklabels=SPECIES,
            yticklabels=SPECIES,
        )
        ax.tick_params(axis="x", labelrotation=25)
        for actual in LABELS:
            for predicted in LABELS:
                count = int(matrix[actual, predicted])
                color = "white" if count > max_count / 2 else "black"
                ax.text(predicted, actual, str(count), ha="center", va="center", color=color)
    fig.suptitle("Held-out Iris test set: rows are actual, columns are predicted")
    fig.savefig(REPORTS / "test_confusion_matrices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    # Fisher's Iris dataset is bundled with scikit-learn; DESCR documents its
    # source and measurements. Use only petal length and width, in that order.
    iris = load_iris()
    description = iris.DESCR
    assert "Iris plants dataset" in description and "R.A. Fisher" in description
    X = iris.data[:, [2, 3]]
    y = iris.target
    assert tuple(iris.feature_names[i] for i in (2, 3)) == (
        "petal length (cm)",
        "petal width (cm)",
    )
    assert tuple(iris.target_names) == SPECIES
    assert X.shape == (150, 2)
    missing = np.isnan(X).sum(axis=0)

    print("IRIS DATASET")
    print("  Source: scikit-learn load_iris() DESCR: Iris plants dataset, R.A. Fisher")
    print(
        "  Tested environment: "
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"Matplotlib {matplotlib.__version__}, scikit-learn {sklearn.__version__}"
    )
    print(f"  Rows: {len(y)} | features: {X.shape[1]}")
    for feature, count in zip((iris.feature_names[2], iris.feature_names[3]), missing):
        print(f"  {feature}: {int(count)} missing values")
    print_partition("all data", np.arange(len(y)), y)

    # Split *indices*, keeping train_test_split's returned order. The test
    # measurements are never used to fit, tune, or scale a development model.
    all_indices = np.arange(len(y))
    development_indices, test_indices = train_test_split(
        all_indices, test_size=0.20, stratify=y, random_state=42
    )
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=0.25,
        stratify=y[development_indices],
        random_state=42,
    )
    groups = [set(part.tolist()) for part in (train_indices, validation_indices, test_indices)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert groups[0] | groups[1] | groups[2] == set(all_indices.tolist())
    print("\nPARTITIONS (disjoint; together cover all 150 rows)")
    print_partition("training", train_indices, y)
    print_partition("validation", validation_indices, y)
    print_partition("test", test_indices, y)

    # All candidates see the same 90 training rows. Each scaler learns only
    # those rows' feature statistics; validation rows are transformed by it.
    validation_rows: list[dict] = []
    for k in K_VALUES:
        candidate = pipeline_for(k)
        candidate.fit(X[train_indices], y[train_indices])
        validation_rows.append(
            {
                "k": k,
                "train_accuracy": accuracy_score(
                    y[train_indices], candidate.predict(X[train_indices])
                ),
                "validation_accuracy": accuracy_score(
                    y[validation_indices], candidate.predict(X[validation_indices])
                ),
            }
        )

    write_csv(
        REPORTS / "knn_validation.csv",
        ["k", "train_accuracy", "validation_accuracy"],
        [
            {
                "k": row["k"],
                "train_accuracy": f'{row["train_accuracy"]:.9f}',
                "validation_accuracy": f'{row["validation_accuracy"]:.9f}',
            }
            for row in validation_rows
        ],
    )
    save_selection_chart(validation_rows)
    print("\nTRAINING-ONLY CANDIDATES")
    print("    k | train accuracy | validation accuracy")
    for row in validation_rows:
        print(f'  {row["k"]:>3d} | {row["train_accuracy"]:.6f}       | {row["validation_accuracy"]:.6f}')

    # Choose on unrounded validation scores; a genuine tie goes to smaller k.
    best = min(validation_rows, key=lambda row: (-row["validation_accuracy"], row["k"]))
    best_k = best["k"]
    print(f"\nSELECTED k={best_k} using validation only (smallest k breaks score ties)")

    # Fit a NEW scaler and KNN on the original development-index order. The
    # baseline sees exactly the same 120 development rows without scaling.
    final_knn = pipeline_for(best_k)
    final_knn.fit(X[development_indices], y[development_indices])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X[development_indices], y[development_indices])

    # One final held-out evaluation stage. Nothing below changes k or settings.
    y_test = y[test_indices]
    predictions = {
        "knn": final_knn.predict(X[test_indices]),
        "baseline": baseline.predict(X[test_indices]),
    }
    summary_rows: list[dict] = []
    per_species_rows: list[dict] = []
    matrices: list[tuple[str, np.ndarray]] = []
    print("\nFINAL HELD-OUT TEST RESULTS (refitted models)")
    for model_name, predicted in predictions.items():
        accuracy = accuracy_score(y_test, predicted)
        macro_f1 = f1_score(y_test, predicted, labels=LABELS, average="macro", zero_division=0)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predicted, labels=LABELS, zero_division=0
        )
        matrix = confusion_matrix(y_test, predicted, labels=LABELS)
        matrices.append((model_name, matrix))
        summary_rows.append(
            {"model": model_name, "accuracy": f"{accuracy:.9f}", "macro_f1": f"{macro_f1:.9f}"}
        )
        print(f"  {model_name}: accuracy={accuracy:.6f}, macro-F1={macro_f1:.6f}")
        print("             species   precision    recall        F1  support")
        for i, species in enumerate(SPECIES):
            per_species_rows.append(
                {
                    "model": model_name,
                    "species": species,
                    "precision": f"{precision[i]:.9f}",
                    "recall": f"{recall[i]:.9f}",
                    "f1": f"{f1[i]:.9f}",
                    "support": int(support[i]),
                }
            )
            print(
                f"  {species:>18}   {precision[i]:.6f}   {recall[i]:.6f}   "
                f"{f1[i]:.6f}      {int(support[i]):2d}"
            )
        print("  Confusion matrix (actual rows, predicted columns: setosa, versicolor, virginica):")
        for row in matrix:
            print("    " + " ".join(f"{int(value):3d}" for value in row))

    write_csv(REPORTS / "test_summary.csv", ["model", "accuracy", "macro_f1"], summary_rows)
    write_csv(
        REPORTS / "test_per_species.csv",
        ["model", "species", "precision", "recall", "f1", "support"],
        per_species_rows,
    )
    save_confusion_charts(matrices, best_k)
    print(f"\nWrote five reports to {REPORTS}")


if __name__ == "__main__":
    main()
