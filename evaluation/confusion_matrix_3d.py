import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# PATHS
# ==================================================

PREDICTIONS_PATH = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\confusion_matrix_3d.png"
)


# ==================================================
# LOAD PREDICTIONS
# ==================================================

print("\n========================================")
print("      LOADING 3D CNN PREDICTIONS")
print("========================================")

if not os.path.exists(PREDICTIONS_PATH):

    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_PATH}"
    )

data = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    "Total predictions:",
    len(data)
)


# ==================================================
# VERIFY REQUIRED COLUMNS
# ==================================================

required_columns = [
    "actual_label",
    "predicted_label"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


# ==================================================
# LABEL CONVERSION
# ==================================================

label_mapping = {
    "REAL": 0,
    "FAKE": 1
}

y_true = data[
    "actual_label"
].map(label_mapping)

y_pred = data[
    "predicted_label"
].map(label_mapping)


if y_true.isna().any():

    raise ValueError(
        "Invalid actual_label values found."
    )

if y_pred.isna().any():

    raise ValueError(
        "Invalid predicted_label values found."
    )


y_true = y_true.astype(int)
y_pred = y_pred.astype(int)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ==================================================
# DISPLAY COUNTS
# ==================================================

print("\n========================================")
print("         3D CNN CONFUSION MATRIX")
print("========================================")

print("\n                Predicted")
print("              REAL    FAKE")

print(
    f"Actual REAL   {tn:4d}    {fp:4d}"
)

print(
    f"Actual FAKE   {fn:4d}    {tp:4d}"
)

print("\n----------------------------------------")

print("True Negative  (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive  (TP):", tp)


# ==================================================
# CREATE GRAPH
# ==================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "FAKE"]
)

display.plot(
    values_format="d"
)

plt.title(
    "3D CNN Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()


# ==================================================
# SAVE GRAPH
# ==================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\nConfusion matrix saved to:")

print(
    OUTPUT_PATH
)

print("\n========================================")
print("       3D CNN CONFUSION MATRIX DONE")
print("========================================")