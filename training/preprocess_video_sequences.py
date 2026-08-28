import os
import numpy as np


# ==================================================
# PATHS
# ==================================================

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"


# ==================================================
# SETTINGS
# ==================================================

EXPECTED_SHAPE = (16, 128, 128, 3)

CLASS_MAPPING = {
    "real": 0,
    "fake": 1
}

SPLITS = ["train", "valid", "test"]


# ==================================================
# VALIDATE ONE SPLIT
# ==================================================

def validate_split(split_name):

    split_path = os.path.join(
        DATASET_PATH,
        split_name
    )

    all_sequences = []
    all_labels = []

    for class_name, label in CLASS_MAPPING.items():

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"Missing folder: {class_path}"
            )

        video_folders = sorted([
            folder
            for folder in os.listdir(class_path)
            if os.path.isdir(
                os.path.join(
                    class_path,
                    folder
                )
            )
        ])

        for video_name in video_folders:

            video_path = os.path.join(
                class_path,
                video_name
            )

            sequence_files = sorted([
                file
                for file in os.listdir(video_path)
                if file.lower().endswith(".npy")
            ])

            for sequence_file in sequence_files:

                sequence_path = os.path.join(
                    video_path,
                    sequence_file
                )

                sequence = np.load(
                    sequence_path,
                    allow_pickle=False
                )

                # ------------------------------------------
                # VERIFY DIMENSION
                # ------------------------------------------

                if sequence.shape != EXPECTED_SHAPE:

                    raise ValueError(
                        f"Invalid shape:\n"
                        f"{sequence_path}\n"
                        f"Expected: {EXPECTED_SHAPE}\n"
                        f"Found: {sequence.shape}"
                    )

                # ------------------------------------------
                # VERIFY DATA TYPE
                # ------------------------------------------

                if sequence.dtype != np.float32:

                    raise ValueError(
                        f"Invalid dtype:\n"
                        f"{sequence_path}\n"
                        f"Expected: float32\n"
                        f"Found: {sequence.dtype}"
                    )

                # ------------------------------------------
                # VERIFY PIXEL RANGE
                # ------------------------------------------

                minimum = float(
                    np.min(sequence)
                )

                maximum = float(
                    np.max(sequence)
                )

                if minimum < 0.0 or maximum > 1.0:

                    raise ValueError(
                        f"Invalid pixel range:\n"
                        f"{sequence_path}\n"
                        f"Min: {minimum}\n"
                        f"Max: {maximum}"
                    )

                # ------------------------------------------
                # STORE
                # ------------------------------------------

                all_sequences.append(sequence)
                all_labels.append(label)

    X = np.array(
        all_sequences,
        dtype=np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int32
    )

    return X, y


# ==================================================
# START
# ==================================================

print("\n========================================")
print("      VIDEO SEQUENCE PREPROCESSING")
print("========================================")


# ==================================================
# TRAIN
# ==================================================

X_train, y_train = validate_split("train")

print("\nTRAIN")
print("Shape :", X_train.shape)
print("Dtype :", X_train.dtype)
print("Min   :", np.min(X_train))
print("Max   :", np.max(X_train))

print(
    "REAL labels:",
    np.sum(y_train == 0)
)

print(
    "FAKE labels:",
    np.sum(y_train == 1)
)


# ==================================================
# VALIDATION
# ==================================================

X_valid, y_valid = validate_split("valid")

print("\nVALIDATION")
print("Shape :", X_valid.shape)
print("Dtype :", X_valid.dtype)
print("Min   :", np.min(X_valid))
print("Max   :", np.max(X_valid))

print(
    "REAL labels:",
    np.sum(y_valid == 0)
)

print(
    "FAKE labels:",
    np.sum(y_valid == 1)
)


# ==================================================
# TEST
# ==================================================

X_test, y_test = validate_split("test")

print("\nTEST")
print("Shape :", X_test.shape)
print("Dtype :", X_test.dtype)
print("Min   :", np.min(X_test))
print("Max   :", np.max(X_test))

print(
    "REAL labels:",
    np.sum(y_test == 0)
)

print(
    "FAKE labels:",
    np.sum(y_test == 1)
)


# ==================================================
# FINAL VERIFICATION
# ==================================================

print("\n========================================")
print("       PREPROCESSING SUMMARY")
print("========================================")

print("Expected shape :", EXPECTED_SHAPE)
print("Pixel range    : 0.0 to 1.0")
print("Label mapping  : REAL=0, FAKE=1")

print("\nTRAIN :", X_train.shape, y_train.shape)
print("VALID :", X_valid.shape, y_valid.shape)
print("TEST  :", X_test.shape, y_test.shape)

print("\nNo additional normalization required.")
print("All sequence preprocessing checks passed.")

print("========================================")