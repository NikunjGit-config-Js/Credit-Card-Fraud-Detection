"""Credit card fraud detection with a leakage-free, imbalance-aware evaluation.

Corrected version of Credit_Card_Fraud_Detection.ipynb, which is kept unchanged as the historical
notebook. Same data, features and model families; corrected methodology:
  * the data is split first (stratified train / validation / test) and the scalers are fitted on the
    training split only;
  * the balanced experiment undersamples the TRAINING split only;
  * every model is scored on the untouched, naturally imbalanced validation split; the model with the
    best validation PR-AUC is then scored once on the untouched test split;
  * fraud-class precision, recall, F1 (default threshold), PR-AUC, ROC-AUC and the confusion matrix.

Usage:
    python fraud_detection.py --data path/to/creditcard.csv
Writes results/validation_metrics.csv, results/test_metrics.json and results/run_info.json.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC

SEED = 1
RESULTS = Path("results")


def split(df):
    """Stratified split with the historical notebook's sizes: 22,000 test and 22,807 validation rows."""
    X, y = df.drop(columns="Class"), df["Class"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=22000, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=22807, stratify=y_dev,
                                                      random_state=SEED)
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale(train, *others):
    """RobustScaler for Amount and min-max for Time, both fitted on the training split only."""
    amount = RobustScaler().fit(train[["Amount"]])
    time = MinMaxScaler().fit(train[["Time"]])
    scaled = []
    for part in (train, *others):
        part = part.copy()
        part["Amount"] = amount.transform(part[["Amount"]]).ravel()
        part["Time"] = time.transform(part[["Time"]]).ravel()
        scaled.append(part.to_numpy(dtype="float32"))
    return scaled


def undersample(X, y):
    """Keep every fraud and an equal-sized random sample of non-fraud rows (training data only)."""
    rng = np.random.default_rng(SEED)
    fraud = np.flatnonzero(y == 1)
    normal = rng.choice(np.flatnonzero(y == 0), size=len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, normal]))
    return X[keep], y[keep]


class ShallowNN:
    """The notebook's Keras network: Dense(units, relu) -> BatchNorm -> Dense(1, sigmoid)."""

    def __init__(self, units, epochs):
        self.units, self.epochs = units, epochs

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(SEED)
        self.model = tf.keras.Sequential([
            tf.keras.Input((X.shape[1],)),
            tf.keras.layers.Dense(self.units, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.model.fit(X, y, epochs=self.epochs, verbose=0)
        return self

    def score(self, X):
        return self.model.predict(X, verbose=0).ravel()


def score(model, X):
    """Continuous fraud score (probability or decision value) and the default-threshold prediction."""
    if isinstance(model, ShallowNN):
        s = model.score(X)
        return s, (s > 0.5).astype(int)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], model.predict(X)
    return model.decision_function(X), model.predict(X)


def metrics(y, s, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
        "f1": f1_score(y, pred), "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def models(balanced):
    """The notebook's model families. Experiment 2 also has the one-unit network and depth-2 boosting."""
    specs = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "shallow_nn_2_units": ShallowNN(2, epochs=80 if balanced else 5),
        "random_forest_depth2": RandomForestClassifier(max_depth=2, n_jobs=-1, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,
                                                        max_depth=2 if balanced else 1, random_state=0),
        "linear_svm_balanced": LinearSVC(class_weight="balanced", random_state=SEED),
    }
    if balanced:
        specs["shallow_nn_1_unit"] = ShallowNN(1, epochs=80)
    return specs


def main():
    parser = argparse.ArgumentParser(description="Credit card fraud detection (corrected evaluation)")
    parser.add_argument("--data", type=Path, default=Path("creditcard.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
    x_train, x_val, x_test = scale(X_train, X_val, X_test)
    y_train, y_val, y_test = (v.to_numpy() for v in (y_train, y_val, y_test))
    training_sets = {"imbalanced": (x_train, y_train), "balanced_undersampled": undersample(x_train, y_train)}

    rows, fitted = [], {}
    for experiment, (X, y) in training_sets.items():
        for name, model in models(balanced=experiment != "imbalanced").items():
            model.fit(X, y)
            rows.append({"experiment": experiment, "model": name, "train_rows": len(y),
                         "train_frauds": int(y.sum()), **metrics(y_val, *score(model, x_val))})
            fitted[(experiment, name)] = model
            print(f"{experiment:22s} {name:22s} validation PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)

    validation = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    validation.round(4).to_csv(RESULTS / "validation_metrics.csv", index=False)

    best = validation.sort_values("pr_auc", ascending=False).iloc[0]
    test = metrics(y_test, *score(fitted[(best["experiment"], best["model"])], x_test))
    (RESULTS / "test_metrics.json").write_text(json.dumps({
        "selected_on": "validation PR-AUC", "experiment": best["experiment"], "model": best["model"],
        "test": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in test.items()},
    }, indent=2) + "\n")

    import tensorflow as tf
    (RESULTS / "run_info.json").write_text(json.dumps({
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(df), "frauds": int(df["Class"].sum()), "seed": SEED,
        "split_rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "split_frauds": {"train": int(y_train.sum()), "validation": int(y_val.sum()), "test": int(y_test.sum())},
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__, "tensorflow": tf.__version__,
    }, indent=2) + "\n")
    print("\nSelected on validation PR-AUC:", best["experiment"], best["model"])
    print("Test:", json.dumps(test))


if __name__ == "__main__":
    main()
