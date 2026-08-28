import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ==================================================
# PATHS
# ==================================================

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"
HISTORY_PATH = r"E:\streesheild\training\training_3d_history.json"


# ==================================================
# SETTINGS
# ==================================================

INPUT_SHAPE = (16, 128, 128, 3)

EPOCHS = 30
BATCH_SIZE = 2

LEARNING_RATE = 0.001


# ==================================================
# LOAD DATA
# ==================================================

def load_split(split_name):

    split_path = os.path.join(
        DATASET_PATH,
        split_name
    )

    sequences = []
    labels = []

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
                        f"Invalid sequence shape: "
                        f"{sequence_path}\n"
                        f"Expected: {INPUT_SHAPE}\n"
                        f"Found: {sequence.shape}"
                    )

                sequences.append(
                    sequence.astype(np.float32)
                )

                labels.append(label)

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.float32
    )

    return X, y


# ==================================================
# LOAD TRAINING DATA
# ==================================================

print("\n========================================")
print("       LOADING 3D CNN DATA")
print("========================================")

X_train, y_train = load_split("train")
X_valid, y_valid = load_split("valid")

print("\nTRAIN")
print("Shape :", X_train.shape)
print("REAL  :", np.sum(y_train == 0))
print("FAKE  :", np.sum(y_train == 1))

print("\nVALIDATION")
print("Shape :", X_valid.shape)
print("REAL  :", np.sum(y_valid == 0))
print("FAKE  :", np.sum(y_valid == 1))


# ==================================================
# CREATE TF.DATA DATASETS
# ==================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=42,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


valid_dataset = tf.data.Dataset.from_tensor_slices(
    (X_valid, y_valid)
)

valid_dataset = valid_dataset.batch(
    BATCH_SIZE
)

valid_dataset = valid_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==================================================
# BUILD FINAL 3D CNN
# ==================================================

model = models.Sequential([

    layers.Input(
        shape=INPUT_SHAPE
    ),

    # ------------------------------------------------
    # CONVOLUTION BLOCK 1
    # ------------------------------------------------

    layers.Conv3D(
        32,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # ------------------------------------------------
    # CONVOLUTION BLOCK 2
    # ------------------------------------------------

    layers.Conv3D(
        64,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # ------------------------------------------------
    # CONVOLUTION BLOCK 3
    # ------------------------------------------------

    layers.Conv3D(
        128,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # ------------------------------------------------
    # FEATURE VECTOR
    # ------------------------------------------------

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        64,
        activation="relu"
    ),

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==================================================
# MODEL INFORMATION
# ==================================================

print("\n========================================")
print("        FINAL 3D CNN CONFIGURATION")
print("========================================")

print(
    "Input shape:",
    INPUT_SHAPE
)

print(
    "Trainable parameters:",
    model.count_params()
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Maximum epochs:",
    EPOCHS
)


# ==================================================
# CALLBACKS
# ==================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)


# ==================================================
# TRAIN
# ==================================================

print("\n========================================")
print("        STARTING FINAL 3D CNN")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# ==================================================
# RESULTS
# ==================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


print("\n========================================")
print("        3D CNN TRAINING COMPLETE")
print("========================================")

print(
    "\nEpochs actually trained:",
    len(train_accuracy)
)

print(
    "Best training accuracy:",
    max(train_accuracy)
)

print(
    "Best validation accuracy:",
    max(val_accuracy)
)

print(
    "Best training loss:",
    min(train_loss)
)

print(
    "Best validation loss:",
    min(val_loss)
)

print(
    "\nFinal training accuracy:",
    train_accuracy[-1]
)

print(
    "Final validation accuracy:",
    val_accuracy[-1]
)

print(
    "Final training loss:",
    train_loss[-1]
)

print(
    "Final validation loss:",
    val_loss[-1]
)


# ==================================================
# SAVE HISTORY
# ==================================================

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )


print(
    "\nTraining history saved to:"
)

print(HISTORY_PATH)

print("\n========================================")
# ==================================================
# SAVE TRAINED 3D CNN FOR EVALUATION
# ==================================================

MODEL_PATH = r"E:\streesheild\models\trained_3d_cnn.keras"

model.save(MODEL_PATH)

print("\nTrained 3D CNN saved to:")
print(MODEL_PATH)