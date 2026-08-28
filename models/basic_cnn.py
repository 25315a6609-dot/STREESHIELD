import tensorflow as tf
from tensorflow.keras import layers, models


# --------------------------------------------------
# BUILD BASIC CNN
# --------------------------------------------------

model = models.Sequential([

    # Input layer
    layers.Input(shape=(128, 128, 3)),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten feature maps
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.5),

    # Output layer
    # 0 → REAL
    # 1 → FAKE
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


print("CNN architecture created successfully.")


# --------------------------------------------------
# COMPILE CNN
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("CNN compiled successfully.")
# --------------------------------------------------
# DISPLAY MODEL SUMMARY
# --------------------------------------------------

print("\n========== CNN MODEL SUMMARY ==========\n")

model.summary()
# --------------------------------------------------
# SAVE CNN MODEL
# --------------------------------------------------

MODEL_PATH = r"E:\streesheild\models\basic_cnn.keras"

model.save(MODEL_PATH)

print("\nCNN model saved successfully.")
print("Saved to:", MODEL_PATH)