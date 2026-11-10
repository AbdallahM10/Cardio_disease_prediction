"""Train a NEW cardiovascular classifier and evaluate it at threshold 0.37.

Prepared by Abdallah Mestrah. Supervisor: Dr. Hussein Kanaan.
Run: python train_and_evaluate.py
Keep this script beside assets/DataSets/cardio.csv.
The dataset path is independent of the terminal working directory.
Requires: numpy, pandas, scikit-learn (see requirements.txt).

No saved model is loaded or overwritten. Each run creates a fresh instance.
Only the selected preprocessing and model are run: no model comparisons,
cross-validation, seed comparisons or threshold searches.
The test partition reproduces the project's previously evaluated partition;
retraining does not make it an independently unseen dataset for the study.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


def main():
    started = perf_counter()
    print("\n[1/6] Load and check the dataset", flush=True)
    dataset = Path(__file__).resolve().parent / "assets/DataSets/cardio.csv"
    if not dataset.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}\n"
            "Keep this script beside the included assets folder."
        )
    print(f"Dataset: {dataset}")
    df = pd.read_csv(dataset)
    features = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active"]
    required = ["id", *features, "cardio"]
    if set(df.columns) != set(required):
        raise ValueError("Dataset columns do not match the project's 11 inputs, ID and target.")
    if df[required].isna().any().any():
        raise ValueError("Unexpected missing values: this pipeline expects the complete cardio dataset.")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite measurements.")
    if not df.id.is_unique:
        raise ValueError("Record IDs must be unique.")
    allowed = {"gender": {1, 2}, "cholesterol": {1, 2, 3}, "gluc": {1, 2, 3},
               "smoke": {0, 1}, "alco": {0, 1}, "active": {0, 1}, "cardio": {0, 1}}
    for column, values in allowed.items():
        if not df[column].isin(values).all():
            raise ValueError(f"Unexpected category in {column}.")
    print(f"Loaded {len(df):,} records. No missing cells or invalid categories.")
    print("Inputs: 11 features. ID excluded; cardio is the target.")

    print("\n[2/6] Remove pressure-inconsistent records", flush=True)
    valid_pressure = (df.ap_hi > 0) & (df.ap_lo > 0) & (df.ap_hi > df.ap_lo)
    clean = df.loc[valid_pressure].copy()
    print(f"Excluded {(~valid_pressure).sum():,}; retained {len(clean):,} records.")
    print("Rule: both pressures positive; systolic greater than diastolic.")

    print("\n[3/6] Separate development and test records", flush=True)
    # Exact-input matches stay together to avoid duplicates crossing the split.
    groups = clean.groupby(features, sort=False).ngroup()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(clean, groups=groups))
    train, test = clean.iloc[train_idx], clean.iloc[test_idx]
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    X_train, y_train = train[features].copy(), train.cardio.copy()
    X_test, y_test = test[features].copy(), test.cardio.copy()
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Both partitions must contain both target classes.")
    print(f"Development: {len(train):,} records. Test: {len(test):,} records.")
    print(f"Development disease share: {y_train.mean():.2%}.")
    print("One fixed split (seed 42); no cross-validation or seed search.")

    print("\n[4/6] Fit IQR caps on development data, then transform both sets", flush=True)
    outlier_columns = ["height", "weight", "ap_hi", "ap_lo"]
    q1 = X_train[outlier_columns].quantile(0.25)
    q3 = X_train[outlier_columns].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    boundaries = pd.DataFrame({"Lower cap": lower, "Upper cap": upper})
    for column in outlier_columns:
        boundaries.loc[column, "Development values capped"] = int(
            (~X_train[column].between(lower[column], upper[column])).sum())
        boundaries.loc[column, "Test values capped"] = int(
            (~X_test[column].between(lower[column], upper[column])).sum())
        X_train[column] = X_train[column].clip(lower[column], upper[column]).astype(float)
        X_test[column] = X_test[column].clip(lower[column], upper[column]).astype(float)
        assert X_train[column].between(lower[column], upper[column]).all()
        assert X_test[column].between(lower[column], upper[column]).all()
    print(boundaries.to_string())
    print("Rows are retained. Test data never determines the caps.")
    print("No scaling or imputation is required by this selected pipeline.")

    print("\n[5/6] Create a NEW model instance and train it", flush=True)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )
    print(model)
    print(f"Fitting on all {len(X_train):,} development records...", flush=True)
    fitting_started = perf_counter()
    model.fit(X_train, y_train)
    assert model.n_iter_ == 200
    print(f"Completed {model.n_iter_} iterations in {perf_counter() - fitting_started:.2f} seconds.")

    threshold = 0.37
    print(f"\n[6/6] Evaluate on test records at threshold {threshold:.2f}", flush=True)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Macro F1": f1_score(y_test, predictions, average="macro"),
        "Disease recall": recall_score(y_test, predictions, pos_label=1),
        "No-disease recall": recall_score(y_test, predictions, pos_label=0),
        "Disease precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "No-disease precision": precision_score(y_test, predictions, pos_label=0, zero_division=0),
    }
    for name, value in scores.items():
        formatted = f"{value:.4f}" if name == "ROC-AUC" else f"{value:.2%}"
        print(f"  {name:24s} {formatted}")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("\nConfusion matrix (rows = actual; columns = predicted):")
    print(pd.DataFrame(matrix, index=["Actual no disease", "Actual disease"],
                       columns=["Predicted no disease", "Predicted disease"]).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["No disease", "Disease"], digits=4, zero_division=0))
    tn, fp, fn, tp = matrix.ravel()
    assert tn + fp + fn + tp == len(test)
    assert np.isclose(scores["Balanced accuracy"],
                      (scores["Disease recall"] + scores["No-disease recall"]) / 2)
    print(f"Correct disease detections: {tp:,}; missed disease cases: {fn:,}.")
    print(f"Correct no-disease predictions: {tn:,}; false alarms: {fp:,}.")
    print(f"\nFinished in {perf_counter() - started:.2f} seconds.")
    print("Fresh model trained in memory. No model or result files loaded or overwritten.")


if __name__ == "__main__":
    main()
