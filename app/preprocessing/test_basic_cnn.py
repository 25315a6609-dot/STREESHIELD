import os
import cv2
import numpy as np
import tensorflow as tf


# ==================================================
# CHANGE THIS PATH
# ==================================================

IMAGE_PATH = r"D:\STREESHIELD_Dataset\processed\test\real\00001.jpg"

MODEL_PATH = (
    r"E:\streesheild\models\trained_basic_cnn.keras"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==================================================
# LOAD IMAGE
# ==================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise ValueError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


print("\n========================================")
print("       BASIC CNN DIRECT TEST")
print("========================================")

print("Image:", IMAGE_PATH)

print(
    "Original shape:",
    image.shape
)


# ==================================================
# PREPROCESS EXACTLY LIKE PHASE 5
# ==================================================

image = cv2.resize(
    image,
    (128, 128)
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = image.astype(
    np.float32
) / 255.0

image = np.expand_dims(
    image,
    axis=0
)


# ==================================================
# PREDICT
# ==================================================

probability = float(
    model.predict(
        image,
        verbose=0
    )[0][0]
)


if probability >= 0.5:

    label = "FAKE"
    confidence = probability * 100

else:

    label = "REAL"
    confidence = (1 - probability) * 100


# ==================================================
# RESULT
# ==================================================

print(
    "\nProbability:",
    probability
)

print(
    "Prediction:",
    label
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("========================================")
