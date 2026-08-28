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

BASIC_PREDICTIONS = (
    r"E:\streesheild\evaluation\test_predictions.csv"
)

CNN3D_PREDICTIONS = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)

BASIC_OUTPUT = (
    r"E:\streesheild\evaluation\phase10_basic_cnn_confusion_matrix.png"
)

CNN3D_OUTPUT = (
    r"E:\streesheild\evaluation\phase10_3d_cnn_confusion_matrix.png"
)


# ==================================================
# LABEL MAPPING
# ==================================================

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1
}


# ==================================================
# CREATE MATRIX
# ==================================================

def create_confusion_matrix(
    prediction_file,
    model_name,
    output_file
):

    if not os.path.exists(prediction_file):

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{prediction_file}"
        )

    data = pd.read_csv(
        prediction_file
    )

    required_columns = [
        "actual_label",
        "predicted_label"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Missing column '{column}' "
                f"in {prediction_file}"
            )

    y_true = (
        data["actual_label"]
        .map(LABEL_MAP)
        .astype(int)
    )

    y_pred = (
        data["predicted_label"]
        .map(LABEL_MAP)
        .astype(int)
    )

    # ----------------------------------------------
    # CREATE MATRIX
    # ----------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n========================================")
    print(
        f"   {model_name} — CONFUSION MATRIX"
    )
    print("========================================")

    print("\n                Predicted")
    print("              REAL    FAKE")

    print(
        f"Actual REAL   {tn:4d}    {fp:4d}"
    )

    print(
        f"Actual FAKE   {fn:4d}    {tp:4d}"
    )

    print("\nTrue Negative  (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)
    print("True Positive  (TP):", tp)

    # ----------------------------------------------
    # PLOT
    # ----------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "REAL",
            "FAKE"
        ]
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved to:")
    print(output_file)

    return cm


# ==================================================
# MAIN
# ==================================================

print("\n========================================")
print(" PHASE 10C — FINAL CONFUSION MATRICES")
print("========================================")


# --------------------------------------------------
# BASIC CNN
# --------------------------------------------------

basic_cm = create_confusion_matrix(
    BASIC_PREDICTIONS,
    "BASIC CNN",
    BASIC_OUTPUT
)


# --------------------------------------------------
# 3D CNN
# --------------------------------------------------

cnn3d_cm = create_confusion_matrix(
    CNN3D_PREDICTIONS,
    "3D CNN",
    CNN3D_OUTPUT
)


# ==================================================
# VERIFY OUTPUT FILES
# ==================================================

print("\n========================================")
print("       OUTPUT VERIFICATION")
print("========================================")

print(
    "Basic CNN matrix exists:",
    os.path.exists(BASIC_OUTPUT)
)

print(
    "3D CNN matrix exists   :",
    os.path.exists(CNN3D_OUTPUT)
)


# ==================================================
# FINAL STATUS
# ==================================================

if (
    os.path.exists(BASIC_OUTPUT)
    and os.path.exists(CNN3D_OUTPUT)
):

    print("\nSTATUS: CONFUSION MATRICES GENERATED")

else:

    print("\nSTATUS: CONFUSION MATRIX GENERATION FAILED")


print("\n========================================")
print("       PHASE 10C COMPLETED")
print("========================================")