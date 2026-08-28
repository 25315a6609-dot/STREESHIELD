import os


# ==================================================
# VERIFIED RESULTS
# ==================================================

basic = {
    "accuracy": 0.6734,
    "precision": 0.6944,
    "recall": 0.6250,
    "f1": 0.6579,
    "roc_auc": 0.7407
}

cnn3d = {
    "accuracy": 0.5000,
    "precision": 0.5000,
    "recall": 1.0000,
    "f1": 0.6667,
    "roc_auc": 0.4700
}


# ==================================================
# CONFUSION MATRIX RESULTS
# ==================================================

basic_cm = {
    "TN": None,
    "FP": None,
    "FN": None,
    "TP": None
}

cnn3d_cm = {
    "TN": 0,
    "FP": 10,
    "FN": 0,
    "TP": 10
}


# ==================================================
# COMPARE METRICS
# ==================================================

basic_wins = 0
cnn3d_wins = 0

for metric in basic:

    if basic[metric] > cnn3d[metric]:

        basic_wins += 1

    elif cnn3d[metric] > basic[metric]:

        cnn3d_wins += 1


# ==================================================
# DETERMINE OVERALL MODEL
# ==================================================

if basic_wins > cnn3d_wins:

    overall_winner = "Basic CNN"

elif cnn3d_wins > basic_wins:

    overall_winner = "3D CNN"

else:

    overall_winner = "Tie"


# ==================================================
# DISPLAY ANALYSIS
# ==================================================

print("\n========================================")
print("       PHASE 8G — RESULTS ANALYSIS")
print("========================================")


# ==================================================
# BASIC CNN
# ==================================================

print("\n========== BASIC CNN ==========")

print(
    "Accuracy :",
    f"{basic['accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{basic['precision'] * 100:.2f}%"
)

print(
    "Recall   :",
    f"{basic['recall'] * 100:.2f}%"
)

print(
    "F1-score :",
    f"{basic['f1'] * 100:.2f}%"
)

print(
    "ROC-AUC  :",
    f"{basic['roc_auc'] * 100:.2f}%"
)


# ==================================================
# 3D CNN
# ==================================================

print("\n========== 3D CNN ==========")

print(
    "Accuracy :",
    f"{cnn3d['accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{cnn3d['precision'] * 100:.2f}%"
)

print(
    "Recall   :",
    f"{cnn3d['recall'] * 100:.2f}%"
)

print(
    "F1-score :",
    f"{cnn3d['f1'] * 100:.2f}%"
)

print(
    "ROC-AUC  :",
    f"{cnn3d['roc_auc'] * 100:.2f}%"
)


# ==================================================
# METRIC WINNERS
# ==================================================

print("\n========== METRIC COMPARISON ==========")

metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1"),
    ("ROC-AUC", "roc_auc")
]

for display_name, metric in metrics:

    if basic[metric] > cnn3d[metric]:

        winner = "Basic CNN"

    elif cnn3d[metric] > basic[metric]:

        winner = "3D CNN"

    else:

        winner = "Tie"

    print(
        f"{display_name:10s}: {winner}"
    )


# ==================================================
# OVERALL WINNER
# ==================================================

print("\n========== OVERALL RESULT ==========")

print(
    "Basic CNN metric wins :",
    basic_wins
)

print(
    "3D CNN metric wins    :",
    cnn3d_wins
)

print(
    "Overall baseline winner:",
    overall_winner
)


# ==================================================
# FALSE PREDICTION ANALYSIS
# ==================================================

print("\n========== FALSE PREDICTION ANALYSIS ==========")

print("\nBasic CNN:")

print(
    "- The Basic CNN makes both REAL and FAKE predictions."
)

print(
    "- Its precision is 69.44%, meaning many predicted FAKE samples are correct."
)

print(
    "- Its recall is 62.50%, so it misses some actual FAKE samples."
)

print(
    "- Its ROC-AUC of 74.07% indicates useful discrimination between the classes."
)


print("\n3D CNN:")

print(
    "- The 3D CNN predicts all 20 test sequences as FAKE."
)

print(
    "- Therefore it detects all 10 FAKE samples."
)

print(
    "- However, it also incorrectly labels all 10 REAL samples as FAKE."
)

print(
    "- This produces 100% recall but only 50% accuracy and 50% precision."
)

print(
    "- ROC-AUC of 47.00% indicates poor ranking/discrimination on this test set."
)


# ==================================================
# STRENGTHS
# ==================================================

print("\n========== STRENGTHS ==========")

print("\nBasic CNN strengths:")

print(
    "- Better overall baseline performance."
)

print(
    "- Higher accuracy, precision, and ROC-AUC."
)

print(
    "- Requires only individual face/image input."
)

print(
    "- Simpler and computationally cheaper than a 3D CNN."
)


print("\n3D CNN strengths:")

print(
    "- Uses temporal information from multiple frames."
)

print(
    "- Can model spatial and temporal patterns jointly."
)

print(
    "- Achieved 100% recall on this particular test set."
)


# ==================================================
# LIMITATIONS
# ==================================================

print("\n========== LIMITATIONS ==========")

print("\nBasic CNN limitations:")

print(
    "- Operates on individual images and does not explicitly model temporal information."
)

print(
    "- A single image can miss temporal inconsistencies that appear across frames."
)


print("\n3D CNN limitations:")

print(
    "- The current dataset contains only 80 training sequences."
)

print(
    "- The 3D CNN therefore has limited training diversity."
)

print(
    "- The model collapsed to predicting all test sequences as FAKE."
)

print(
    "- 3D CNN evaluation uses 20 video sequences, while the Basic CNN uses 398 image samples."
)

print(
    "- Therefore the comparison is useful as a baseline but is not a perfectly controlled apples-to-apples experiment."
)


# ==================================================
# WHY BASIC CNN PERFORMED BETTER
# ==================================================

print("\n========== INTERPRETATION ==========")

print(
    "The Basic CNN performed better in this experiment primarily because "
    "it was trained/evaluated with substantially more image-level examples "
    "than the 3D CNN had video-sequence examples."
)

print(
    "The 3D CNN has the potential to capture temporal information, "
    "but the current video dataset is too small for reliable generalization."
)

print(
    "The current results should therefore be treated as a baseline comparison, "
    "not evidence that image-based CNNs are universally better than 3D CNNs."
)


# ==================================================
# SAVE TEXT ANALYSIS
# ==================================================

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\phase8_analysis.txt"
)

analysis_text = f"""
PHASE 8G — RESULTS ANALYSIS

Overall baseline winner: {overall_winner}

Basic CNN:
Accuracy: {basic['accuracy'] * 100:.2f}%
Precision: {basic['precision'] * 100:.2f}%
Recall: {basic['recall'] * 100:.2f}%
F1-score: {basic['f1'] * 100:.2f}%
ROC-AUC: {basic['roc_auc'] * 100:.2f}%

3D CNN:
Accuracy: {cnn3d['accuracy'] * 100:.2f}%
Precision: {cnn3d['precision'] * 100:.2f}%
Recall: {cnn3d['recall'] * 100:.2f}%
F1-score: {cnn3d['f1'] * 100:.2f}%
ROC-AUC: {cnn3d['roc_auc'] * 100:.2f}%

Basic CNN metric wins: {basic_wins}
3D CNN metric wins: {cnn3d_wins}

The Basic CNN is the stronger baseline in this experiment.
The 3D CNN has temporal modeling capability but generalized poorly
on the small video-sequence test set.
"""

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        analysis_text.strip()
    )


# ==================================================
# FINAL STATUS
# ==================================================

print("\n========================================")
print("       PHASE 8G COMPLETED")
print("========================================")

print(
    "Analysis saved to:"
)

print(
    OUTPUT_PATH
)

print("========================================")