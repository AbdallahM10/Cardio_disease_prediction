# Cardiovascular Disease Prediction

A university machine-learning project by **Abdallah Mestrah**, supervised by **Dr. Hussein Kanaan**.

Predicting the dataset's cardiovascular disease label from 11 patient measurements and categorical inputs. The selected classifier is scikit-learn's **Histogram Gradient Boosting**, with IQR clipping and a **0.37 decision threshold**.

## Interactive demo

Start with `START_HERE.txt` for installation and launch instructions. On Windows, after installing the root requirements in `.venv`, double-click `interface/start_app.bat`. The app uses the included saved model and needs no dataset. It preserves the original **0.50** threshold and its baseline metrics; the training script uses **0.37**.

## Contents

- `train_and_evaluate.py`: preprocesses data, trains a new model, and prints evaluation metrics. It does not load or overwrite the saved model.
- `cardiovascular_disease_report.ipynb`: exploration, model comparisons, evaluation, and threshold analysis, including saved outputs.
- `Cardiovascular_Study_Editorial.html`: standalone presentation; open it in a browser. Use arrow keys to navigate.
- `assets/`: the saved baseline model and supporting evaluation records required by the notebook.
- `requirements.txt`: Python dependencies.

## Run locally

Use Python 3.12. From this folder:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python train_and_evaluate.py
```

The dataset is intentionally not published here: its redistribution license has not been confirmed. Place the original, unmodified project CSV at `assets/DataSets/cardio.csv` before running the script or notebook. The CSV must match the dataset recorded in `assets/source_manifest.json`; a different file may produce different results or fail the notebook's integrity check. It contains 70,000 rows and columns `id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio`.

To use the notebook, run `jupyter notebook`, open the report, and select this environment's Python kernel. Saved results can be viewed without rerunning it. Only load the bundled Joblib artifact from a trusted copy of this repository.

## Method

1. Exclude ID from the inputs and remove inconsistent blood pressure rows.
2. Reserve approximately 20% of identical-input groups for testing, keeping identical inputs together to prevent overlap.
3. Learn IQR clipping bounds on development data only and apply those bounds to test inputs.
4. Train Histogram Gradient Boosting: 200 iterations, learning rate 0.03, 15 maximum leaf nodes, and 20 minimum samples per leaf.
5. Classify disease when its predicted score is at least 0.37.

The report also documents five-fold grouped validation and historical comparisons. The standalone script performs one training/evaluation run; it does not rerun those searches.

## Results at the selected threshold

On the project's 13,748-row reserved test partition:

| Metric | Result |
|---|---:|
| Accuracy | 71.79% |
| Disease recall / sensitivity | 81.60% |
| No-disease recall / specificity | 62.38% |
| Disease precision | 67.56% |
| Macro F1 | 71.60% |
| ROC-AUC | 0.8051 |

Lowering the threshold detects more disease cases and also produces more false alarms. The saved model bundle and baseline evaluation assets retain the original **0.50** threshold; the script and the notebook's final threshold section use **0.37**. The learned trees are unchanged by threshold selection. Historical 0.50 results remain explicitly identified in the presentation and report.

## Limits

This is an educational classification project, not a clinical diagnostic tool or a prediction of future disease risk. The test partition was examined during the study and threshold comparisons; these results are not an independent external validation. Predicted scores have not been established as calibrated clinical probabilities. No performance improvement is guaranteed on other datasets.

The full experiment folders, environments, submission ZIP, and build files are deliberately excluded from this repository. No license is granted for the externally supplied dataset.
