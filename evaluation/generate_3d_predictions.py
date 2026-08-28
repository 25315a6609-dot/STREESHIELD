import os
import csv
import numpy as np
import tensorflow as tf


# ==================================================
# PATHS
# ==================================================

MODEL_PATH = r"E:\streesheild\models\trained_3d_cnn.keras"

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"

OUTPUT_PATH = (
    r"E:\streesheild\evaluation\test_3d_predictions.csv"
)


# ==================================================
# SETTINGS
# ==================================================

INPUT_SHAPE = (16, 128, 128, 3)


# ==================================================
# LOAD TEST DATA WITH FILENAMES
# ==================================================

def load_test_data():

    test_path = os.path.join(
        DATASET_PATH,
        "test"
    )

    sequences = []
    labels = []
    filenames = []

    class_mapping = {
        "real": 0,
        "fake": 1
    }

    for class_name, label in class_mapping.items():

        class_path = os.path.join(
            test_path,
            class_name
        )

        for video_name in sorted(
            os.listdir(class_path)
        ):

            video_path = os.path.join(
                class_path,
                video_name
            )

            if not os.path.isdir(video_path):
                continue

            for sequence_file in sorted(
                os.listdir(video_path)
            ):

                if not sequence_file.lower().endswith(".npy"):
                    continue

                sequence_path = os.path.join(
                    video_path,
                    sequence_file
                )

                sequence = np.load(
                    sequence_path,
                    allow_pickle=False
                )

                if sequence.shape != INPUT_SHAPE:

                    raise ValueError(
                        f"Invalid shape:\n"
                        f"{sequence_path}\n"
                        f"Expected: {INPUT_SHAPE}\n"
                        f"Found: {sequence.shape}"
                    )

                sequences.append(
                    sequence.astype(np.float32)
                )

                labels.append(label)

                filenames.append(
                    os.path.join(
                        class_name,
                        video_name,
                        sequence_file
                    )
                )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        filenames
    )


# ==================================================
# LOAD MODEL
# ==================================================

print("\n========================================")
print("       LOADING TRAINED 3D CNN")
print("========================================")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==================================================
# LOAD TEST DATA
# ==================================================

X_test, y_test, filenames = load_test_data()

print("\n========================================")
print("          TEST DATA")
print("========================================")

print("Total sequences:", len(X_test))

print(
    "REAL:",
    np.sum(y_test == 0)
)

print(
    "FAKE:",
    np.sum(y_test == 1)
)


# ==================================================
# GENERATE PROBABILITIES
# ==================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=1
).flatten()


# ==================================================
# CONVERT TO LABELS
# ==================================================

predicted_labels = (
    probabilities >= 0.5
).astype(np.int32)


# ==================================================
# SAVE CSV
# ==================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "actual_label",
        "predicted_probability",
        "predicted_label"
    ])

    for filename, actual, probability, predicted in zip(
        filenames,
        y_test,
        probabilities,
        predicted_labels
    ):

        actual_name = (
            "FAKE"
            if actual == 1
            else "REAL"
        )

        predicted_name = (
            "FAKE"
            if predicted == 1
            else "REAL"
        )

        writer.writerow([
            filename,
            actual_name,
            f"{probability:.6f}",
            predicted_name
        ])


# ==================================================
# SUMMARY
# ==================================================

predicted_real = np.sum(
    predicted_labels == 0
)

predicted_fake = np.sum(
    predicted_labels == 1
)


print("\n========================================")
print("       3D CNN PREDICTION SUMMARY")
print("========================================")

print(
    "Total predictions :",
    len(predicted_labels)
)

print(
    "Predicted REAL    :",
    predicted_real
)

print(
    "Predicted FAKE    :",
    predicted_fake
)

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print("========================================")