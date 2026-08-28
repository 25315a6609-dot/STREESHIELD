import os
import matplotlib.pyplot as plt


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR = r"E:\streesheild\evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# VERIFIED METRICS
# ==================================================

metrics = {
    "Accuracy": {
        "Basic CNN": 0.6734,
        "3D CNN": 0.5000
    },

    "Precision": {
        "Basic CNN": 0.6944,
        "3D CNN": 0.5000
    },

    "Recall": {
        "Basic CNN": 0.6250,
        "3D CNN": 1.0000
    },

    "F1-score": {
        "Basic CNN": 0.6579,
        "3D CNN": 0.6667
    }
}


# ==================================================
# CREATE ONE GRAPH PER METRIC
# ==================================================

for metric_name, values in metrics.items():

    models = list(values.keys())
    scores = list(values.values())

    plt.figure(figsize=(7, 5))

    plt.bar(
        models,
        scores
    )

    plt.ylim(0, 1)

    plt.ylabel("Score")

    plt.xlabel("Model")

    plt.title(
        f"{metric_name} Comparison"
    )

    # Display percentages above bars
    for index, score in enumerate(scores):

        plt.text(
            index,
            score + 0.02,
            f"{score * 100:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    # Convert metric name to filename
    filename = (
        metric_name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_comparison.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"{metric_name} graph saved to:"
    )

    print(
        output_path
    )


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========================================")
print("       PHASE 8F COMPLETED")
print("========================================")

print(
    "Created graphs:"
)

print(
    "1. Accuracy comparison"
)

print(
    "2. Precision comparison"
)

print(
    "3. Recall comparison"
)

print(
    "4. F1-score comparison"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("========================================")