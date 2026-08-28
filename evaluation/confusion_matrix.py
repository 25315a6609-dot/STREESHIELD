import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------
# PATH
# --------------------------------------------------

PREDICTIONS_PATH = r"E:\streesheild\evaluation\test_predictions.csv"

OUTPUT_PATH = r"E:\streesheild\evaluation\confusion_matrix.png"


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


# --------------------------------------------------
# CREATE CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# --------------------------------------------------
# DISPLAY COUNTS
# --------------------------------------------------

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print("\n                Predicted")
print("              REAL    FAKE")
print(f"Actual REAL   {tn:4d}    {fp:4d}")
print(f"Actual FAKE   {fn:4d}    {tp:4d}")

print("\n----------------------------------------")

print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")


# --------------------------------------------------
# PLOT CONFUSION MATRIX
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "FAKE"]
)

display.plot()

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nConfusion matrix graph saved to:")
print(OUTPUT_PATH)

print("========================================")