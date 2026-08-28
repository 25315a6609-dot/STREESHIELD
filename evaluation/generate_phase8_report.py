import os
from datetime import datetime


# ==================================================
# PATHS
# ==================================================

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\phase8_final_report.txt"
)


# ==================================================
# VERIFIED RESULTS
# ==================================================

basic = {
    "accuracy": 67.34,
    "precision": 69.44,
    "recall": 62.50,
    "f1": 65.79,
    "roc_auc": 74.07
}

cnn3d = {
    "accuracy": 50.00,
    "precision": 50.00,
    "recall": 100.00,
    "f1": 66.67,
    "roc_auc": 47.00
}


# ==================================================
# CONFUSION MATRICES
# ==================================================

basic_cm = {
    "TN": "Available in basic_cnn_confusion_matrix.png",
    "FP": "Available in basic_cnn_confusion_matrix.png",
    "FN": "Available in basic_cnn_confusion_matrix.png",
    "TP": "Available in basic_cnn_confusion_matrix.png"
}

cnn3d_cm = {
    "TN": 0,
    "FP": 10,
    "FN": 0,
    "TP": 10
}


# ==================================================
# OUTPUT FILES
# ==================================================

artifacts = [
    r"E:\streesheild\models\trained_basic_cnn.keras",
    r"E:\streesheild\models\trained_3d_cnn.keras",
    r"E:\streesheild\evaluation\test_predictions.csv",
    r"E:\streesheild\evaluation\test_3d_predictions.csv",
    r"E:\streesheild\evaluation\model_prediction_comparison.csv",
    r"E:\streesheild\evaluation\basic_cnn_confusion_matrix.png",
    r"E:\streesheild\evaluation\cnn3d_confusion_matrix.png",
    r"E:\streesheild\evaluation\accuracy_comparison.png",
    r"E:\streesheild\evaluation\precision_comparison.png",
    r"E:\streesheild\evaluation\recall_comparison.png",
    r"E:\streesheild\evaluation\f1_score_comparison.png",
    r"E:\streesheild\evaluation\model_performance_comparison.csv",
    r"E:\streesheild\evaluation\phase8_analysis.txt"
]


# ==================================================
# DETERMINE OVERALL WINNER
# ==================================================

basic_wins = 3
cnn3d_wins = 2

overall_winner = "Basic CNN"


# ==================================================
# GENERATE REPORT
# ==================================================

report = f"""
============================================================
              STREESHIELD — PHASE 8 REPORT
              CNN vs 3D CNN COMPARISON
============================================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


============================================================
1. OBJECTIVE
============================================================

Phase 8 compares two deepfake-detection approaches:

Basic CNN:
Image-based detection using individual face images.

3D CNN:
Video-based detection using sequences of 16 frames.


============================================================
2. MODELS
============================================================

Basic CNN model:
E:\\streesheild\\models\\trained_basic_cnn.keras

3D CNN model:
E:\\streesheild\\models\\trained_3d_cnn.keras


============================================================
3. PERFORMANCE COMPARISON
============================================================

Metric                 Basic CNN          3D CNN
------------------------------------------------------------
Accuracy               {basic["accuracy"]:.2f}%              {cnn3d["accuracy"]:.2f}%
Precision              {basic["precision"]:.2f}%              {cnn3d["precision"]:.2f}%
Recall                 {basic["recall"]:.2f}%             {cnn3d["recall"]:.2f}%
F1-score               {basic["f1"]:.2f}%              {cnn3d["f1"]:.2f}%
ROC-AUC                {basic["roc_auc"]:.2f}%              {cnn3d["roc_auc"]:.2f}%


============================================================
4. TEST DATA
============================================================

Basic CNN:
398 test image predictions
198 REAL
200 FAKE

3D CNN:
20 test sequence predictions
10 REAL
10 FAKE

Important:
The two models were evaluated using different sample types
and different test-set sizes. Therefore this is a baseline
comparison rather than a perfectly controlled apples-to-apples
experiment.


============================================================
5. CONFUSION MATRICES
============================================================

Basic CNN:
See:
E:\\streesheild\\evaluation\\basic_cnn_confusion_matrix.png

3D CNN:

                Predicted
              REAL    FAKE
Actual REAL      0      10
Actual FAKE      0      10

TN = 0
FP = 10
FN = 0
TP = 10


============================================================
6. PREDICTION BEHAVIOR
============================================================

Basic CNN:

The Basic CNN produced both REAL and FAKE predictions.
It achieved 67.34% accuracy and 69.44% precision.

Its recall was 62.50%, meaning some actual FAKE images
were missed.

Its ROC-AUC was 74.07%, showing useful class-ranking ability.

3D CNN:

The 3D CNN predicted all 20 test sequences as FAKE.

This resulted in:

Accuracy  = 50.00%
Precision = 50.00%
Recall    = 100.00%
F1-score  = 66.67%
ROC-AUC   = 47.00%

The 100% recall should therefore not be interpreted as
strong overall performance because all REAL sequences were
also classified as FAKE.


============================================================
7. METRIC WINNERS
============================================================

Accuracy   -> Basic CNN
Precision  -> Basic CNN
Recall     -> 3D CNN
F1-score   -> 3D CNN
ROC-AUC    -> Basic CNN

Basic CNN metric wins = 3
3D CNN metric wins    = 2


============================================================
8. OVERALL WINNER
============================================================

OVERALL BASELINE WINNER: BASIC CNN


Reason:

The Basic CNN achieved higher Accuracy, Precision, and
ROC-AUC and produced substantially more balanced predictions.

The 3D CNN achieved higher Recall and a slightly higher
F1-score, but this occurred because it classified every
test sequence as FAKE.


============================================================
9. BASIC CNN STRENGTHS
============================================================

1. Better overall baseline accuracy.
2. Higher precision.
3. Higher ROC-AUC.
4. Produces both REAL and FAKE predictions.
5. Simpler image-based inference.
6. Requires fewer temporal resources than a 3D CNN.


============================================================
10. BASIC CNN LIMITATIONS
============================================================

1. Processes individual images rather than complete video
   temporal information.
2. May miss inconsistencies that only become visible across
   multiple consecutive frames.
3. Its recall of 62.50% means some FAKE samples were missed.


============================================================
11. 3D CNN STRENGTHS
============================================================

1. Processes multiple frames simultaneously.
2. Can learn spatial and temporal features jointly.
3. Has the potential to detect temporal inconsistencies.
4. Achieved 100% recall on this particular test set.


============================================================
12. 3D CNN LIMITATIONS
============================================================

1. The current training dataset contains only 80 training
   sequences.
2. The model generalized poorly to the validation and test
   sets.
3. It predicted all test sequences as FAKE.
4. ROC-AUC was below random-chance performance in this
   experiment.
5. 3D CNN training is more computationally expensive.
6. The current comparison uses only 20 video test sequences.


============================================================
13. FINAL INTERPRETATION
============================================================

The Basic CNN is the stronger model in the current
STREESHIELD baseline experiment.

The result does NOT mean that Basic CNNs are inherently
better than 3D CNNs for deepfake detection.

Instead, the result indicates that the current 3D CNN
did not have sufficient video-sequence training diversity
to generalize reliably.

The 3D CNN successfully demonstrated that the architecture
can learn from the sequences during the overfitting sanity
test, but it failed to generalize on the held-out test set.

A larger video dataset, more independent source videos,
more temporal sequences per video, and stronger video-model
training would be required for a reliable comparison.


============================================================
14. GENERATED ARTIFACTS
============================================================
"""


# ==================================================
# ADD ARTIFACT STATUS
# ==================================================

for artifact in artifacts:

    exists = os.path.exists(artifact)

    status = "EXISTS" if exists else "MISSING"

    report += (
        f"\n{status:8s}  {artifact}"
    )


# ==================================================
# FINAL CONCLUSION
# ==================================================

report += f"""


============================================================
15. PHASE 8 CONCLUSION
============================================================

Phase 8 comparison is complete.

Basic CNN:
67.34% Accuracy
69.44% Precision
62.50% Recall
65.79% F1-score
74.07% ROC-AUC

3D CNN:
50.00% Accuracy
50.00% Precision
100.00% Recall
66.67% F1-score
47.00% ROC-AUC

Overall baseline winner:
BASIC CNN

============================================================
                  END OF PHASE 8
============================================================
"""


# ==================================================
# SAVE REPORT
# ==================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report.strip()
    )


# ==================================================
# VERIFY
# ==================================================

print("\n========================================")
print("     PHASE 8H — FINAL REPORT")
print("========================================")

print(
    "Report saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_PATH)
)

if os.path.exists(OUTPUT_PATH):

    print(
        "File size:",
        os.path.getsize(OUTPUT_PATH),
        "bytes"
    )

print("\n========================================")
print("       PHASE 8H COMPLETED")
print("========================================")