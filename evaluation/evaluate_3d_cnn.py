import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# PATHS
# ==================================================

MODEL_PATH = r"E:\streesheild\models\trained_3d_cnn.keras"

DATASET_PATH = (
    r"D:\STREESHIELD_VideoDataset\split_sequences"
)


# ==================================================
# SETTINGS
# ==================================================

INPUT_SHAPE = (16, 128, 128, 3)

BATCH_SIZE = 2


# ==================================================
# LOAD TEST DATA
# ==================================================

def load_test_data():

    test_path = os.path.join(
        DATASET_PATH,
        "test"
    )

    sequences = []
    labels = []

    # REAL = 0
    # FAKE = 1

    class_mapping = {
        "real": 0,
        "fake": 1
    }

    for class_name, label in class_mapping.items():

        class_path = os.path.join(
            test_path,
            class_name
        )

        if not os.path.exists(class_path):

            raise FileNotFoundError(
                f"Missing folder: {class_path}"
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

                if not sequence_file.lower().endswith(
                    ".npy"
                ):
                    continue

                sequence_path = os.path.join(
                    video_path,
                    sequence_file
                )

                sequence = np.load(
                    sequence_path,
                    allow_pickle=False
                )

                # ------------------------------------------
                # VERIFY SHAPE
                # ------------------------------------------

                if sequence.shape != INPUT_SHAPE:

                    raise ValueError(
                        f"Invalid sequence shape:\n"
                        f"{sequence_path}\n"
                        f"Expected: {INPUT_SHAPE}\n"
                        f"Found: {sequence.shape}"
                    )

                sequences.append(
                    sequence.astype(np.float32)
                )

                labels.append(label)

    X_test = np.asarray(
        sequences,
        dtype=np.float32
    )

    y_test = np.asarray(
        labels,
        dtype=np.int32
    )

    return X_test, y_test


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

print("\n========================================")
print("       LOADING TRAINED 3D CNN")
print("========================================")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    "Model input shape :",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ==================================================
# LOAD TEST DATA
# ==================================================

X_test, y_test = load_test_data()

print("\n========================================")
print("          TEST DATASET")
print("========================================")

print(
    "Test shape :",
    X_test.shape
)

print(
    "Labels     :",
    y_test.shape
)

print(
    "REAL       :",
    np.sum(y_test == 0)
)

print(
    "FAKE       :",
    np.sum(y_test == 1)
)


# ==================================================
# EVALUATE MODEL
# ==================================================

print("\n========================================")
print("       TEST DATASET EVALUATION")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ==================================================
# GENERATE PREDICTIONS
# ==================================================

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).flatten()


# Convert probability to class
# >= 0.5 = FAKE
# < 0.5  = REAL

predicted_labels = (
    probabilities >= 0.5
).astype(np.int32)


# ==================================================
# CALCULATE METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    predicted_labels
)

precision = precision_score(
    y_test,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicted_labels,
    zero_division=0
)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("          3D CNN TEST RESULTS")
print("========================================")

print(
    f"Test loss     : {test_loss:.4f}"
)

print(
    f"Test accuracy : {test_accuracy:.4f} "
    f"({test_accuracy * 100:.2f}%)"
)

print(
    f"Accuracy      : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision     : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall        : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-score      : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print("========================================")