import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==================================================
# PATHS
# ==================================================

BASIC_CNN_PATH = (
    r"E:\streesheild\evaluation\test_predictions.csv"
)

CNN3D_PATH = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)

BASIC_OUTPUT = (
    r"E:\streesheild\evaluation\basic_cnn_confusion_matrix.png"
)

CNN3D_OUTPUT = (
    r"E:\streesheild\evaluation\cnn3d_confusion_matrix.png"
)


# ==================================================
# LABEL CONVERSION
# ==================================================

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1
}


def convert_labels(series):
    return series.map(LABEL_MAP).astype(int)


# ==================================================
# CREATE CONFUSION MATRIX
# ==================================================

def create_matrix(
    prediction_path,
    model_name,
    output_path
):

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_path}"
        )

    data = pd.read_csv(
        prediction_path
    )

    required_columns = [
        "actual_label",
        "predicted_label"
    ]

    for column in required_columns:

        if column not in data.columns:
            raise ValueError(
                f"Missing column '{column}' "
                f"in {prediction_path}"
            )

    y_true = convert_labels(
        data["actual_label"]
    )

    y_pred = convert_labels(
        data["predicted_label"]
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n========================================")
    print(f"       {model_name} CONFUSION MATRIX")
    print("========================================")

    print("\n                Predicted")
    print("              REAL    FAKE")

    print(
        f"Actual REAL   {tn:4d}    {fp:4d}"
    )

    print(
        f"Actual FAKE   {fn:4d}    {tp:4d}"
    )

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    # ------------------------------------------------
    # PLOT
    # ------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
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
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved to:",
        output_path
    )

    return cm


# ==================================================
# BASIC CNN
# ==================================================

print("\n========================================")
print("      PHASE 8D — CONFUSION MATRICES")
print("========================================")

basic_cm = create_matrix(
    BASIC_CNN_PATH,
    "BASIC CNN",
    BASIC_OUTPUT
)


# ==================================================
# 3D CNN
# ==================================================

cnn3d_cm = create_matrix(
    CNN3D_PATH,
    "3D CNN",
    CNN3D_OUTPUT
)


# ==================================================
# VERIFY FILES
# ==================================================

print("\n========================================")
print("       CONFUSION MATRIX SUMMARY")
print("========================================")

print(
    "Basic CNN matrix exists:",
    os.path.exists(BASIC_OUTPUT)
)

print(
    "3D CNN matrix exists  :",
    os.path.exists(CNN3D_OUTPUT)
)

print("\n========================================")
print("       PHASE 8D COMPLETED")
print("========================================")