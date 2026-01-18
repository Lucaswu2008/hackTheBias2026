"""Train a lightweight KNN classifier from landmark CSV data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KNN classifier from landmark CSV data.")
    parser.add_argument("--data", default="data/landmarks.csv", help="CSV file with labels and features.")
    parser.add_argument("--model", default="models/asl_knn.npz", help="Output model path.")
    parser.add_argument("--labels", default="models/labels.json", help="Output labels path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors.")
    return parser.parse_args()


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    labels = []
    features = []
    with path.open("r", newline="") as handle:
        header = handle.readline().strip().split(",")
        has_header = header and header[0].lower() == "label"
        if not has_header:
            handle.seek(0)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            labels.append(parts[0])
            features.append([float(x) for x in parts[1:]])

    if not labels:
        raise ValueError("No samples found in the data file.")

    return np.array(features, dtype=np.float32), np.array(labels)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def knn_predict_single(
    x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int
) -> tuple[str, float]:
    dists = np.linalg.norm(X_train - x, axis=1)
    k = max(1, min(k, len(dists)))
    idx = np.argpartition(dists, k - 1)[:k]

    weights = 1.0 / (dists[idx] + 1e-6)
    scores: dict[str, float] = {}
    for label, weight in zip(y_train[idx], weights):
        scores[label] = scores.get(label, 0.0) + float(weight)

    top_label = max(scores.items(), key=lambda item: item[1])[0]
    conf = scores[top_label] / max(sum(scores.values()), 1e-6)
    return top_label, float(conf)


def evaluate(X: np.ndarray, y: np.ndarray, k: int) -> None:
    correct = 0
    total = len(X)
    per_label_total: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()

    for x, label in zip(X, y):
        pred, _ = knn_predict_single(x, X, y, k)
        per_label_total[label] += 1
        if pred == label:
            correct += 1
            per_label_correct[label] += 1

    accuracy = correct / max(total, 1)
    print(f"Accuracy (self-eval): {accuracy:.2%}")

    labels = sorted(per_label_total.keys())
    for label in labels:
        total_label = per_label_total[label]
        correct_label = per_label_correct[label]
        rate = correct_label / max(total_label, 1)
        print(f"{label}: {rate:.2%} ({correct_label}/{total_label})")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    X, y = load_data(data_path)

    if len(set(y)) < 2:
        print("Need at least two classes to train a classifier.")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, args.test_size, args.seed)
    mean, scale = standardize_fit(X_train)
    X_train_scaled = standardize_apply(X_train, mean, scale)
    X_test_scaled = standardize_apply(X_test, mean, scale)

    print("Evaluation on held-out data:")
    correct = 0
    for x, label in zip(X_test_scaled, y_test):
        pred, _ = knn_predict_single(x, X_train_scaled, y_train, args.k)
        if pred == label:
            correct += 1
    accuracy = correct / max(len(X_test_scaled), 1)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(X_test_scaled)})")

    print("Training set per-label accuracy:")
    evaluate(X_train_scaled, y_train, args.k)

    full_mean, full_scale = standardize_fit(X)
    X_scaled = standardize_apply(X, full_mean, full_scale)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        model_path,
        X=X_scaled.astype(np.float32),
        y=y.astype("U"),
        mean=full_mean.astype(np.float32),
        scale=full_scale.astype(np.float32),
        k=np.array([args.k], dtype=np.int32),
    )

    classes = sorted(set(y.tolist()))
    labels_path.write_text(json.dumps(classes, indent=2), encoding="ascii")

    print(f"Saved model to: {model_path}")
    print(f"Saved labels to: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
