import json
import os


# --------------------------------------------------
# PATHS
# --------------------------------------------------

HISTORY_PATH = r"E:\streesheild\training\training_history.json"
RESULTS_PATH = r"E:\streesheild\evaluation\final_results.json"

TRAINED_MODEL_PATH = r"E:\streesheild\models\trained_basic_cnn.keras"
PREDICTIONS_PATH = r"E:\streesheild\evaluation\test_predictions.csv"
CONFUSION_MATRIX_PATH = r"E:\streesheild\evaluation\confusion_matrix.png"


# --------------------------------------------------
# LOAD TRAINING HISTORY
# --------------------------------------------------

with open(HISTORY_PATH, "r") as file:
    history = json.load(file)


# --------------------------------------------------
# FINAL TRAINING VALUES
# --------------------------------------------------

final_training_accuracy = history["accuracy"][-1]
final_validation_accuracy = history["val_accuracy"][-1]

final_training_loss = history["loss"][-1]
final_validation_loss = history["val_loss"][-1]


# --------------------------------------------------
# TEST RESULTS
# --------------------------------------------------

test_accuracy = 0.6734
test_loss = 0.6455


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

precision = 0.6944
recall = 0.6250
f1_score = 0.6579
roc_auc = 0.7407


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

true_negative = 143
false_positive = 55
false_negative = 75
true_positive = 125


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

dataset_information = {
    "train_real": 980,
    "train_fake": 997,
    "valid_real": 196,
    "valid_fake": 200,
    "test_real": 198,
    "test_fake": 200,
    "total_processed_images": 2771,
    "image_size": "128x128",
    "image_format": "JPEG"
}


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

final_results = {

    "model": {
        "name": "Basic CNN",
        "input_shape": "128x128x3",
        "optimizer": "Adam",
        "loss": "Binary Crossentropy",
        "output_activation": "Sigmoid",
        "trained_model": TRAINED_MODEL_PATH
    },

    "training": {
        "epochs": len(history["accuracy"]),
        "final_training_accuracy": final_training_accuracy,
        "final_validation_accuracy": final_validation_accuracy,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss
    },

    "test": {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss
    },

    "metrics": {
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc
    },

    "confusion_matrix": {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive
    },

    "dataset": dataset_information,

    "files": {
        "training_history": HISTORY_PATH,
        "predictions": PREDICTIONS_PATH,
        "confusion_matrix": CONFUSION_MATRIX_PATH,
        "trained_model": TRAINED_MODEL_PATH
    }
}


# --------------------------------------------------
# SAVE FINAL RESULTS
# --------------------------------------------------

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_results,
        file,
        indent=4
    )


# --------------------------------------------------
# DISPLAY FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("       STREESHIELD PHASE 5 RESULTS")
print("========================================")

print("\nTRAINING")
print(f"Training accuracy   : {final_training_accuracy * 100:.2f}%")
print(f"Validation accuracy : {final_validation_accuracy * 100:.2f}%")
print(f"Training loss       : {final_training_loss:.4f}")
print(f"Validation loss     : {final_validation_loss:.4f}")

print("\nTEST")
print(f"Test accuracy       : {test_accuracy * 100:.2f}%")
print(f"Test loss           : {test_loss:.4f}")

print("\nPERFORMANCE METRICS")
print(f"Precision           : {precision * 100:.2f}%")
print(f"Recall              : {recall * 100:.2f}%")
print(f"F1-score            : {f1_score * 100:.2f}%")
print(f"ROC-AUC             : {roc_auc:.4f}")

print("\nCONFUSION MATRIX")
print(f"True Negative       : {true_negative}")
print(f"False Positive      : {false_positive}")
print(f"False Negative      : {false_negative}")
print(f"True Positive       : {true_positive}")

print("\n----------------------------------------")
print("Final results saved to:")
print(RESULTS_PATH)

print("\nTrained model:")
print(TRAINED_MODEL_PATH)

print("========================================")