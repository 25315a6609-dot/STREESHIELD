import pandas as pd
import os


# ==================================================
# OUTPUT
# ==================================================

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\model_performance_comparison.csv"
)


# ==================================================
# VERIFIED METRICS FROM PHASE 8B
# ==================================================

metrics = {
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],

    "Basic CNN": [
        0.6734,
        0.6944,
        0.6250,
        0.6579,
        0.7407
    ],

    "3D CNN": [
        0.5000,
        0.5000,
        1.0000,
        0.6667,
        0.4700
    ]
}


# ==================================================
# CREATE TABLE
# ==================================================

comparison = pd.DataFrame(
    metrics
)


# ==================================================
# BEST MODEL BY METRIC
# ==================================================

best_models = []

for _, row in comparison.iterrows():

    basic_value = row["Basic CNN"]
    cnn3d_value = row["3D CNN"]

    if basic_value > cnn3d_value:

        best = "Basic CNN"

    elif cnn3d_value > basic_value:

        best = "3D CNN"

    else:

        best = "Tie"

    best_models.append(best)


comparison["Best Model"] = best_models


# ==================================================
# SAVE
# ==================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("       PHASE 8E — PERFORMANCE")
print("========================================")

print("\nPerformance comparison:\n")

print(
    comparison.to_string(
        index=False,
        formatters={
            "Basic CNN": "{:.4f}".format,
            "3D CNN": "{:.4f}".format
        }
    )
)


print("\n========================================")
print("          METRIC WINNERS")
print("========================================")

for _, row in comparison.iterrows():

    print(
        f"{row['Metric']:10s} : "
        f"{row['Best Model']}"
    )


# ==================================================
# OVERALL OBSERVATION
# ==================================================

print("\n========================================")
print("        OVERALL OBSERVATION")
print("========================================")

basic_wins = sum(
    comparison["Best Model"] == "Basic CNN"
)

cnn3d_wins = sum(
    comparison["Best Model"] == "3D CNN"
)

print(
    "Basic CNN metric wins :",
    basic_wins
)

print(
    "3D CNN metric wins    :",
    cnn3d_wins
)

if basic_wins > cnn3d_wins:

    print(
        "\nOverall baseline winner: Basic CNN"
    )

elif cnn3d_wins > basic_wins:

    print(
        "\nOverall baseline winner: 3D CNN"
    )

else:

    print(
        "\nOverall result: Tie"
    )


# ==================================================
# SAVE CHECK
# ==================================================

print("\nComparison table saved to:")
print(OUTPUT_PATH)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_PATH)
)

print("\n========================================")
print("       PHASE 8E COMPLETED")
print("========================================")