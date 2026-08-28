import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# PATHS
# ==================================================

BASIC_CNN_PREDICTIONS = (
    r"E:\streesheild\evaluation\test_predictions.csv"
)

CNN3D_PREDICTIONS = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def find_column(df, candidates):
    """
    Find a column using several possible names.
    """

    lower_map = {
        column.lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in lower_map:
            return lower_map[key]

    return None


def convert_label(value):
    """
    Convert REAL/FAKE or 0/1 to numeric labels.

    REAL = 0
    FAKE = 1
    """

    if isinstance(value, str):

        value = value.strip().upper()

        if value == "REAL":
            return 0

        if value == "FAKE":
            return 1

    return int(value)


def load_prediction_file(path, model_name):

    print("\n========================================")
    print(f"      LOADING {model_name}")
    print("========================================")

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print("File:", path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # ------------------------------------------------
    # FIND ACTUAL LABEL COLUMN
    # ------------------------------------------------

    actual_column = find_column(
        df,
        [
            "actual_label",
            "actual",
            "true_label",
            "true",
            "label",
            "y_true"
        ]
    )

    # ------------------------------------------------
    # FIND PREDICTED LABEL COLUMN
    # ------------------------------------------------

    predicted_column = find_column(
        df,
        [
            "predicted_label",
            "prediction",
            "predicted",
            "y_pred"
        ]
    )

    # ------------------------------------------------
    # FIND PROBABILITY / SCORE COLUMN
    # ------------------------------------------------

    probability_column = find_column(
        df,
        [
            "predicted_probability",
            "probability",
            "prob",
            "score",
            "confidence",
            "prediction_probability",
            "y_score"
        ]
    )

    if actual_column is None:
        raise ValueError(
            f"Could not find actual-label column in {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    if predicted_column is None:
        raise ValueError(
            f"Could not find predicted-label column in {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    if probability_column is None:
        raise ValueError(
            f"Could not find probability/score column in {path}\n"
            f"ROC-AUC requires a probability or prediction score.\n"
            f"Available columns: {list(df.columns)}"
        )

    print("\nDetected columns:")
    print("Actual label :", actual_column)
    print("Predicted    :", predicted_column)
    print("Probability  :", probability_column)

    # ------------------------------------------------
    # CONVERT LABELS
    # ------------------------------------------------

    y_true = df[actual_column].apply(
        convert_label
    ).to_numpy()

    y_pred = df[predicted_column].apply(
        convert_label
    ).to_numpy()

    y_score = pd.to_numeric(
        df[probability_column],
        errors="coerce"
    ).to_numpy()

    if pd.isna(y_score).any():

        raise ValueError(
            f"Invalid probability values found in {path}"
        )

    return y_true, y_pred, y_score


# ==================================================
# CALCULATE METRICS
# ==================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_score
):

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_score
        )
    }


# ==================================================
# LOAD BASIC CNN
# ==================================================

basic_true, basic_pred, basic_score = load_prediction_file(
    BASIC_CNN_PREDICTIONS,
    "BASIC CNN"
)


# ==================================================
# LOAD 3D CNN
# ==================================================

cnn3d_true, cnn3d_pred, cnn3d_score = load_prediction_file(
    CNN3D_PREDICTIONS,
    "3D CNN"
)


# ==================================================
# VERIFY LABEL COUNTS
# ==================================================

print("\n========================================")
print("          DATASET CHECK")
print("========================================")

print("\nBASIC CNN")

print(
    "REAL actual:",
    sum(basic_true == 0)
)

print(
    "FAKE actual:",
    sum(basic_true == 1)
)

print(
    "REAL predicted:",
    sum(basic_pred == 0)
)

print(
    "FAKE predicted:",
    sum(basic_pred == 1)
)


print("\n3D CNN")

print(
    "REAL actual:",
    sum(cnn3d_true == 0)
)

print(
    "FAKE actual:",
    sum(cnn3d_true == 1)
)

print(
    "REAL predicted:",
    sum(cnn3d_pred == 0)
)

print(
    "FAKE predicted:",
    sum(cnn3d_pred == 1)
)


# ==================================================
# CALCULATE
# ==================================================

basic_metrics = calculate_metrics(
    basic_true,
    basic_pred,
    basic_score
)

cnn3d_metrics = calculate_metrics(
    cnn3d_true,
    cnn3d_pred,
    cnn3d_score
)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("       PHASE 8B — MODEL EVALUATION")
print("========================================")


print("\nBASIC CNN")

print(
    f"Accuracy  : "
    f"{basic_metrics['accuracy']:.4f} "
    f"({basic_metrics['accuracy'] * 100:.2f}%)"
)

print(
    f"Precision : "
    f"{basic_metrics['precision']:.4f} "
    f"({basic_metrics['precision'] * 100:.2f}%)"
)

print(
    f"Recall    : "
    f"{basic_metrics['recall']:.4f} "
    f"({basic_metrics['recall'] * 100:.2f}%)"
)

print(
    f"F1-score  : "
    f"{basic_metrics['f1']:.4f} "
    f"({basic_metrics['f1'] * 100:.2f}%)"
)

print(
    f"ROC-AUC   : "
    f"{basic_metrics['roc_auc']:.4f} "
    f"({basic_metrics['roc_auc'] * 100:.2f}%)"
)


print("\n3D CNN")

print(
    f"Accuracy  : "
    f"{cnn3d_metrics['accuracy']:.4f} "
    f"({cnn3d_metrics['accuracy'] * 100:.2f}%)"
)

print(
    f"Precision : "
    f"{cnn3d_metrics['precision']:.4f} "
    f"({cnn3d_metrics['precision'] * 100:.2f}%)"
)

print(
    f"Recall    : "
    f"{cnn3d_metrics['recall']:.4f} "
    f"({cnn3d_metrics['recall'] * 100:.2f}%)"
)

print(
    f"F1-score  : "
    f"{cnn3d_metrics['f1']:.4f} "
    f"({cnn3d_metrics['f1'] * 100:.2f}%)"
)

print(
    f"ROC-AUC   : "
    f"{cnn3d_metrics['roc_auc']:.4f} "
    f"({cnn3d_metrics['roc_auc'] * 100:.2f}%)"
)


print("\n========================================")
print("       PHASE 8B COMPLETED")
print("========================================")