import os
import pandas as pd


# ==================================================
# PATHS
# ==================================================

BASIC_CNN_PATH = (
    r"E:\streesheild\evaluation\test_predictions.csv"
)

CNN3D_PATH = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\model_prediction_comparison.csv"
)


# ==================================================
# LOAD BASIC CNN
# ==================================================

print("\n========================================")
print("       LOADING MODEL PREDICTIONS")
print("========================================")

if not os.path.exists(BASIC_CNN_PATH):
    raise FileNotFoundError(
        f"Basic CNN predictions not found:\n{BASIC_CNN_PATH}"
    )

if not os.path.exists(CNN3D_PATH):
    raise FileNotFoundError(
        f"3D CNN predictions not found:\n{CNN3D_PATH}"
    )


basic = pd.read_csv(
    BASIC_CNN_PATH
)

cnn3d = pd.read_csv(
    CNN3D_PATH
)


# ==================================================
# BASIC CNN SUMMARY
# ==================================================

basic_summary = pd.DataFrame({
    "model": ["Basic CNN"] * len(basic),
    "filename": basic["filename"],
    "actual_label": basic["actual_label"],
    "predicted_probability": basic["predicted_probability"],
    "predicted_label": basic["predicted_label"]
})


# ==================================================
# 3D CNN SUMMARY
# ==================================================

cnn3d_summary = pd.DataFrame({
    "model": ["3D CNN"] * len(cnn3d),
    "filename": cnn3d["filename"],
    "actual_label": cnn3d["actual_label"],
    "predicted_probability": cnn3d["predicted_probability"],
    "predicted_label": cnn3d["predicted_label"]
})


# ==================================================
# COMBINE
# ==================================================

comparison = pd.concat(
    [
        basic_summary,
        cnn3d_summary
    ],
    ignore_index=True
)


# ==================================================
# SAVE
# ==================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# DISPLAY SUMMARY
# ==================================================

print("\n========================================")
print("       PHASE 8C — PREDICTIONS")
print("========================================")

print(
    "\nBasic CNN predictions:",
    len(basic_summary)
)

print(
    "3D CNN predictions:",
    len(cnn3d_summary)
)

print(
    "Combined predictions:",
    len(comparison)
)

print("\nBasic CNN predicted labels:")

print(
    basic["predicted_label"]
    .value_counts()
)


print("\n3D CNN predicted labels:")

print(
    cnn3d["predicted_label"]
    .value_counts()
)


print("\nComparison file saved to:")

print(
    OUTPUT_PATH
)

print("\n========================================")
print("       PHASE 8C COMPLETED")
print("========================================")