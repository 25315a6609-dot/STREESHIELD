import os
import numpy as np


# ==================================================
# PATHS
# ==================================================

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"


# ==================================================
# SETTINGS
# ==================================================

SEQUENCE_SHAPE = (16, 128, 128, 3)


# ==================================================
# LOAD ONE SPLIT
# ==================================================

def load_split(split_name):

    split_path = os.path.join(
        DATASET_PATH,
        split_name
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

                if sequence.shape != SEQUENCE_SHAPE:

                    raise ValueError(
                        f"Invalid shape in "
                        f"{sequence_path}: "
                        f"{sequence.shape}"
                    )

                sequences.append(sequence)
                labels.append(label)

    X = np.array(
        sequences,
        dtype=np.float32
    )

    y = np.array(
        labels,
        dtype=np.int32
    )

    return X, y


# ==================================================
# LOAD TRAIN
# ==================================================

print("\n========================================")
print("       LOADING VIDEO SEQUENCES")
print("========================================")

X_train, y_train = load_split("train")

print("\nTRAIN")
print("Sequences :", len(X_train))
print("Shape     :", X_train.shape)
print("Labels    :", y_train.shape)


# ==================================================
# LOAD VALIDATION
# ==================================================

X_valid, y_valid = load_split("valid")

print("\nVALIDATION")
print("Sequences :", len(X_valid))
print("Shape     :", X_valid.shape)
print("Labels    :", y_valid.shape)


# ==================================================
# LOAD TEST
# ==================================================

X_test, y_test = load_split("test")

print("\nTEST")
print("Sequences :", len(X_test))
print("Shape     :", X_test.shape)
print("Labels    :", y_test.shape)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========================================")
print("       SEQUENCE LOADING SUMMARY")
print("========================================")

print("TRAIN :", X_train.shape, y_train.shape)
print("VALID :", X_valid.shape, y_valid.shape)
print("TEST  :", X_test.shape, y_test.shape)

print("\nREAL training labels:",
      np.sum(y_train == 0))

print("FAKE training labels:",
      np.sum(y_train == 1))

print("\nREAL validation labels:",
      np.sum(y_valid == 0))

print("FAKE validation labels:",
      np.sum(y_valid == 1))

print("\nREAL test labels:",
      np.sum(y_test == 0))

print("FAKE test labels:",
      np.sum(y_test == 1))

print("\nAll video sequences loaded successfully.")