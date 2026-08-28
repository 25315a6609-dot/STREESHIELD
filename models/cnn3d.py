import tensorflow as tf
from tensorflow.keras import layers, models


# --------------------------------------------------
# 3D CNN INPUT
# --------------------------------------------------

INPUT_SHAPE = (16, 128, 128, 3)


# --------------------------------------------------
# BUILD 3D CNN
# --------------------------------------------------

model = models.Sequential([

    # Input
    layers.Input(
        shape=INPUT_SHAPE
    ),

    # --------------------------------------------------
    # 3D CONVOLUTION BLOCK 1
    # --------------------------------------------------

    layers.Conv3D(
        32,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # --------------------------------------------------
    # 3D CONVOLUTION BLOCK 2
    # --------------------------------------------------

    layers.Conv3D(
        64,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # --------------------------------------------------
    # 3D CONVOLUTION BLOCK 3
    # --------------------------------------------------

    layers.Conv3D(
        128,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling3D(
        pool_size=(2, 2, 2)
    ),

    # --------------------------------------------------
    # FLATTEN
    # --------------------------------------------------

    layers.Flatten(),

    # --------------------------------------------------
    # DENSE LAYER
    # --------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    # --------------------------------------------------
    # DROPOUT
    # --------------------------------------------------

    layers.Dropout(
        0.5
    ),

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------------------------
# DISPLAY BASIC INFORMATION
# --------------------------------------------------

print("3D CNN architecture created successfully.")

print(
    "Input shape:",
    INPUT_SHAPE
)

print(
    "Output shape:",
    model.output_shape
)
# --------------------------------------------------
# COMPILE 3D CNN
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("3D CNN compiled successfully.")
# --------------------------------------------------
# DISPLAY MODEL SUMMARY
# --------------------------------------------------

print("\n========== 3D CNN MODEL SUMMARY ==========\n")

model.summary()