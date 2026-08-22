import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.classifier.features import extract_features, features_to_vector

DATA_PATH = Path(__file__).parent / "data" / "labeled_prompts.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def load_dataset() -> tuple[list[list[float]], list[int]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m app.classifier.generate_dataset` first."
        )

    X, y = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feats = extract_features(row["prompt"], has_context=bool(int(row["has_context"])))
            X.append(features_to_vector(feats))
            y.append(int(row["tier"]))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    args = parser.parse_args()

    X, y = load_dataset()
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if args.model == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    print(f"Model: {args.model}")
    print(f"Held-out accuracy: {acc:.3f}")
    print("\nConfusion matrix (rows=true tier, cols=predicted tier 1/2/3):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    if acc < 0.80:
        print("WARNING: accuracy below 80% target. Consider --model random_forest "
              "or expanding the labeled dataset.")

    joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
    print(f"\nSaved trained classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
