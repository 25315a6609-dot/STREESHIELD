import json
import os
import matplotlib.pyplot as plt


# ==================================================
# PATHS
# ==================================================

BASIC_HISTORY_PATH = (
    r"E:\streesheild\training\training_history.json"
)

CNN3D_HISTORY_PATH = (
    r"E:\streesheild\training\training_3d_history.json"
)

OUTPUT_DIR = (
    r"E:\streesheild\evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# VERIFIED TEST METRICS
# ==================================================

basic_metrics = {
    "Accuracy": 0.6734,
    "Precision": 0.6944,
    "Recall": 0.6250,
    "F1-score": 0.6579
}

cnn3d_metrics = {
    "Accuracy": 0.5000,
    "Precision": 0.5000,
    "Recall": 1.0000,
    "F1-score": 0.6667
}


# ==================================================
# LOAD TRAINING HISTORY
# ==================================================

def load_history(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"History file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


basic_history = load_history(
    BASIC_HISTORY_PATH
)

cnn3d_history = load_history(
    CNN3D_HISTORY_PATH
)


# ==================================================
# HELPER — BAR GRAPH
# ==================================================

def create_metric_graph(
    metric_name,
    basic_value,
    cnn3d_value,
    filename
):

    labels = [
        "Basic CNN",
        "3D CNN"
    ]

    values = [
        basic_value,
        cnn3d_value
    ]

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        f"{metric_name} Comparison"
    )

    for index, value in enumerate(values):

        plt.text(
            index,
            min(value + 0.02, 0.98),
            f"{value * 100:.2f}%",
            ha="center"
        )

    plt.tight_layout()

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
# ACCURACY
# ==================================================

create_metric_graph(
    "Accuracy",
    basic_metrics["Accuracy"],
    cnn3d_metrics["Accuracy"],
    "phase10_accuracy.png"
)


# ==================================================
# PRECISION
# ==================================================

create_metric_graph(
    "Precision",
    basic_metrics["Precision"],
    cnn3d_metrics["Precision"],
    "phase10_precision.png"
)


# ==================================================
# RECALL
# ==================================================

create_metric_graph(
    "Recall",
    basic_metrics["Recall"],
    cnn3d_metrics["Recall"],
    "phase10_recall.png"
)


# ==================================================
# F1-SCORE
# ==================================================

create_metric_graph(
    "F1-score",
    basic_metrics["F1-score"],
    cnn3d_metrics["F1-score"],
    "phase10_f1_score.png"
)


# ==================================================
# LOSS GRAPH
# ==================================================

basic_loss = basic_history.get(
    "loss",
    []
)

basic_val_loss = basic_history.get(
    "val_loss",
    []
)

cnn3d_loss = cnn3d_history.get(
    "loss",
    []
)

cnn3d_val_loss = cnn3d_history.get(
    "val_loss",
    []
)


if not basic_loss or not cnn3d_loss:

    raise ValueError(
        "Training loss history is missing."
    )


basic_epochs = range(
    1,
    len(basic_loss) + 1
)

cnn3d_epochs = range(
    1,
    len(cnn3d_loss) + 1
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    basic_epochs,
    basic_loss,
    label="Basic CNN - Training Loss"
)

if basic_val_loss:
    plt.plot(
        basic_epochs,
        basic_val_loss,
        label="Basic CNN - Validation Loss"
    )

plt.plot(
    cnn3d_epochs,
    cnn3d_loss,
    label="3D CNN - Training Loss"
)

if cnn3d_val_loss:
    plt.plot(
        cnn3d_epochs,
        cnn3d_val_loss,
        label="3D CNN - Validation Loss"
    )

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss Comparison"
)

plt.legend()

plt.tight_layout()

loss_output = os.path.join(
    OUTPUT_DIR,
    "phase10_loss.png"
)

plt.savefig(
    loss_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nLoss graph saved to:"
)

print(
    loss_output
)


# ==================================================
# FINAL STATUS
# ==================================================

print("\n========================================")
print("       PHASE 10B COMPLETED")
print("========================================")

print(
    "Generated graphs:"
)

print(
    "1. phase10_accuracy.png"
)

print(
    "2. phase10_loss.png"
)

print(
    "3. phase10_precision.png"
)

print(
    "4. phase10_recall.png"
)

print(
    "5. phase10_f1_score.png"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("========================================")