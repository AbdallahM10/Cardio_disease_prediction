"""Inference only: validate inputs, apply saved caps, then predict. Never fit."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/final_evaluation/model_with_clipping.joblib"
FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"]


def load_model():
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError("This saved model requires scikit-learn 1.9.1. Install requirements.txt from the repository root.")
    bundle = joblib.load(MODEL_PATH)
    if bundle["feature_columns"] != FEATURES or bundle["threshold"] != 0.50:
        raise ValueError("The saved model does not match the selected 11-feature, threshold-0.50 pipeline.")
    return bundle


def predict(bundle, values):
    if set(values) != set(FEATURES):
        raise ValueError("All 11 input fields are required.")
    raw = pd.DataFrame([values], columns=bundle["feature_columns"])
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Every measurement must be a finite number.")
    allowed = {"gender": {1, 2}, "cholesterol": {1, 2, 3}, "gluc": {1, 2, 3},
               "smoke": {0, 1}, "alco": {0, 1}, "active": {0, 1}}
    for name, categories in allowed.items():
        if values[name] not in categories:
            raise ValueError(f"Invalid category for {name}.")
    if values["age"] < round(18 * 365.25) or values["age"] > round(120 * 365.25):
        raise ValueError("Enter an adult age between 18 and 120 years for this demonstration.")
    if values["height"] <= 0 or values["weight"] <= 0:
        raise ValueError("Height and weight must be positive.")
    if values["ap_hi"] <= 0 or values["ap_lo"] <= 0:
        raise ValueError("Both blood pressure readings must be positive.")
    if values["ap_hi"] <= values["ap_lo"]:
        raise ValueError("Systolic pressure must be greater than diastolic pressure.")
    processed = raw.copy()
    capped = []
    for name in bundle["outlier_columns"]:
        value = float(raw.at[0, name])
        clipped = float(np.clip(value, bundle["lower_bounds"][name], bundle["upper_bounds"][name]))
        processed[name] = float(clipped)
        if value != clipped:
            capped.append({"Feature": name, "Entered value": value, "Model input": clipped})
    probability = float(bundle["model"].predict_proba(processed)[0, 1])
    return {"disease": probability >= bundle["threshold"], "probability": probability,
            "threshold": bundle["threshold"], "processed": processed, "capped": capped}
