import pandas as pd
import os


# ==================================================
# OUTPUT
# ==================================================

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\phase10_model_comparison.csv"
)


# ==================================================
# FINAL VERIFIED METRICS
# ==================================================

data = {
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

comparison = pd.DataFrame(data)


# ==================================================
# FIND WINNER FOR EACH METRIC
# ==================================================

winners = []

for _, row in comparison.iterrows():

    if row["Basic CNN"] > row["3D CNN"]:

        winners.append(
            "Basic CNN"
        )

    elif row["3D CNN"] > row["Basic CNN"]:

        winners.append(
            "3D CNN"
        )

    else:

        winners.append(
            "Tie"
        )


comparison["Winner"] = winners


# ==================================================
# OVERALL
# ==================================================

basic_wins = winners.count(
    "Basic CNN"
)

cnn3d_wins = winners.count(
    "3D CNN"
)


if basic_wins > cnn3d_wins:

    overall_winner = "Basic CNN"

elif cnn3d_wins > basic_wins:

    overall_winner = "3D CNN"

else:

    overall_winner = "Tie"


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
print("      PHASE 10E — FINAL MODEL COMPARISON")
print("========================================")

print("\n")
print(
    comparison.to_string(
        index=False
    )
)

print("\n========================================")
print("             WINNERS")
print("========================================")

for _, row in comparison.iterrows():

    print(
        f"{row['Metric']:10s} -> "
        f"{row['Winner']}"
    )


print("\nBasic CNN wins:", basic_wins)
print("3D CNN wins   :", cnn3d_wins)

print(
    "\nOverall final model:",
    overall_winner
)

print("\nComparison saved to:")
print(OUTPUT_PATH)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_PATH)
)

print("\n========================================")
print("       PHASE 10E COMPLETED")
print("========================================")