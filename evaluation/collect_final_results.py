import json
import os


# ==================================================
# PATHS
# ==================================================

BASIC_HISTORY_PATH = (
    r"E:\streesheild\training\training_history.json"
)

CNN3D_HISTORY_PATH = (
    r"E:\streesheild\training\training_3d_history.json"
)

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\phase10_final_results.json"
)


# ==================================================
# VERIFIED FINAL TEST METRICS
# ==================================================

basic_test = {
    "accuracy": 0.6734,
    "precision": 0.6944,
    "recall": 0.6250,
    "f1_score": 0.6579,
    "roc_auc": 0.7407
}

cnn3d_test = {
    "accuracy": 0.5000,
    "precision": 0.5000,
    "recall": 1.0000,
    "f1_score": 0.6667,
    "roc_auc": 0.4700
}


# ==================================================
# LOAD TRAINING HISTORIES
# ==================================================

def load_history(path):

    if not os.path.exists(path):

        print(
            f"WARNING: History file not found:\n{path}"
        )

        return {}

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
# BUILD FINAL RESULT OBJECT
# ==================================================

results = {

    "project": "STREESHIELD",

    "basic_cnn": {

        "model_path":
            r"E:\streesheild\models\trained_basic_cnn.keras",

        "test_predictions":
            r"E:\streesheild\evaluation\test_predictions.csv",

        "test_metrics":
            basic_test,

        "training_history":
            basic_history
    },

    "cnn_3d": {

        "model_path":
            r"E:\streesheild\models\trained_3d_cnn.keras",

        "test_predictions":
            r"E:\streesheild\evaluation\test_3d_predictions.csv",

        "test_metrics":
            cnn3d_test,

        "training_history":
            cnn3d_history
    },

    "comparison": {

        "overall_baseline_winner":
            "Basic CNN",

        "accuracy_winner":
            "Basic CNN",

        "precision_winner":
            "Basic CNN",

        "recall_winner":
            "3D CNN",

        "f1_score_winner":
            "3D CNN",

        "roc_auc_winner":
            "Basic CNN"
    }
}


# ==================================================
# SAVE
# ==================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==================================================
# DISPLAY
# ==================================================

print("\n========================================")
print("       PHASE 10A — FINAL RESULTS")
print("========================================")

print("\nBASIC CNN")

for metric, value in basic_test.items():

    print(
        f"{metric:10s}: "
        f"{value * 100:.2f}%"
    )


print("\n3D CNN")

for metric, value in cnn3d_test.items():

    print(
        f"{metric:10s}: "
        f"{value * 100:.2f}%"
    )


print("\nOVERALL BASELINE WINNER:")
print("Basic CNN")


print("\nFinal results saved to:")
print(OUTPUT_PATH)

print(
    "\nFile exists:",
    os.path.exists(OUTPUT_PATH)
)

print("\n========================================")
print("       PHASE 10A COMPLETED")
print("========================================")