import os


# ==================================================
# FINAL CONFUSION MATRIX VALUES
# ==================================================

basic = {
    "TN": 143,
    "FP": 55,
    "FN": 75,
    "TP": 125
}

cnn3d = {
    "TN": 0,
    "FP": 10,
    "FN": 0,
    "TP": 10
}


# ==================================================
# ERROR COUNTS
# ==================================================

print("\n========================================")
print("       PHASE 10F — ERROR ANALYSIS")
print("========================================")


print("\n========== BASIC CNN ==========")

print(
    "False Positives:",
    basic["FP"]
)

print(
    "False Negatives:",
    basic["FN"]
)

print(
    "Total errors:",
    basic["FP"] + basic["FN"]
)


print("\n========== 3D CNN ==========")

print(
    "False Positives:",
    cnn3d["FP"]
)

print(
    "False Negatives:",
    cnn3d["FN"]
)

print(
    "Total errors:",
    cnn3d["FP"] + cnn3d["FN"]
)


# ==================================================
# ERROR INTERPRETATION
# ==================================================

print("\n========================================")
print("          POSSIBLE REASONS")
print("========================================")


print("\nBasic CNN — False Positives")

print(
    "- Some REAL images may contain compression artifacts, "
    "lighting changes, blur, or unusual facial features."
)

print(
    "- Differences between training and real-world input "
    "quality can cause REAL images to be classified as FAKE."
)


print("\nBasic CNN — False Negatives")

print(
    "- Some FAKE images may contain high-quality facial "
    "generation with few visible artifacts."
)

print(
    "- A single image does not provide temporal information."
)

print(
    "- Subtle manipulation artifacts may not be visible "
    "in an individual frame."
)


print("\n3D CNN — False Positives")

print(
    "- All 10 REAL test sequences were classified as FAKE."
)

print(
    "- This indicates strong class bias toward the FAKE class."
)

print(
    "- The small video training dataset is a likely contributor."
)


print("\n3D CNN — False Negatives")

print(
    "- No FAKE sequence was classified as REAL in this test."
)

print(
    "- Therefore FN = 0, but this should not be interpreted "
    "as perfect fake detection because the model overpredicts FAKE."
)


# ==================================================
# OVERALL ANALYSIS
# ==================================================

print("\n========================================")
print("          OVERALL ERROR ANALYSIS")
print("========================================")

print(
    "The Basic CNN produces both false positives and "
    "false negatives, showing a more balanced decision pattern."
)

print(
    "The 3D CNN predicts every test sequence as FAKE, "
    "which explains its 100% recall but only 50% accuracy."
)

print(
    "The 3D CNN requires substantially more video training "
    "data and greater sequence diversity for reliable generalization."
)


# ==================================================
# SAVE REPORT
# ==================================================

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\phase10_error_analysis.txt"
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        """
PHASE 10F — ERROR ANALYSIS

BASIC CNN
TN = 143
FP = 55
FN = 75
TP = 125

3D CNN
TN = 0
FP = 10
FN = 0
TP = 10

BASIC CNN:
False positives occur when REAL images are classified as FAKE.
Possible causes include compression artifacts, lighting,
blur, unusual facial characteristics, and input-distribution
differences.

False negatives occur when FAKE images are classified as REAL.
Possible causes include high-quality manipulations and the lack
of temporal information in single-image analysis.

3D CNN:
The model classified every test sequence as FAKE.
This creates 10 false positives and 0 false negatives.
The behavior indicates strong bias toward the FAKE class and
poor generalization from the small video training dataset.

OVERALL:
The Basic CNN provides a more balanced error pattern.
The 3D CNN requires more diverse video sequences and further
training improvements before it can become a reliable temporal
deepfake detector.
""".strip()
    )


print("\nError analysis saved to:")
print(OUTPUT_PATH)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_PATH)
)

print("\n========================================")
print("       PHASE 10F COMPLETED")
print("========================================")