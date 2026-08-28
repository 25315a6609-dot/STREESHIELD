import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# PATH
# --------------------------------------------------

PREDICTIONS_PATH = r"E:\streesheild\evaluation\test_predictions.csv"


# --------------------------------------------------
# LOAD PREDICTIONS
# --------------------------------------------------

data = pd.read_csv(PREDICTIONS_PATH)

print("Prediction file loaded successfully.")
print("Total test samples:", len(data))


# --------------------------------------------------
# CONVERT LABELS TO NUMBERS
# --------------------------------------------------

y_true = data["actual_label"].map({
    "REAL": 0,
    "FAKE": 1
})

y_pred = data["predicted_label"].map({
    "REAL": 0,
    "FAKE": 1
})

y_probability = data["predicted_probability"]


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========================================")
print("        CNN PERFORMANCE METRICS")
print("========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score  : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("========================================")