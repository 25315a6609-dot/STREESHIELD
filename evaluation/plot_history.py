import json
import os
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

HISTORY_PATH = r"E:\streesheild\training\training_history.json"
OUTPUT_FOLDER = r"E:\streesheild\evaluation"


# --------------------------------------------------
# LOAD TRAINING HISTORY
# --------------------------------------------------

with open(HISTORY_PATH, "r") as file:
    history = json.load(file)


# --------------------------------------------------
# GET HISTORY VALUES
# --------------------------------------------------

train_accuracy = history["accuracy"]
valid_accuracy = history["val_accuracy"]

train_loss = history["loss"]
valid_loss = history["val_loss"]

epochs = range(1, len(train_accuracy) + 1)


# --------------------------------------------------
# ACCURACY GRAPH
# --------------------------------------------------

plt.figure()

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    valid_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    OUTPUT_FOLDER,
    "accuracy_graph.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Accuracy graph saved to:")
print(accuracy_path)


# --------------------------------------------------
# LOSS GRAPH
# --------------------------------------------------

plt.figure()

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    valid_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    OUTPUT_FOLDER,
    "loss_graph.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nLoss graph saved to:")
print(loss_path)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\nTraining graphs generated successfully.")