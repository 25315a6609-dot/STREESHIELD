import os
import tensorflow as tf


# ==================================================
# MODEL PATHS
# ==================================================

BASIC_CNN_PATH = (
    r"E:\streesheild\models\trained_basic_cnn.keras"
)

CNN3D_PATH = (
    r"E:\streesheild\models\trained_3d_cnn.keras"
)


# ==================================================
# CHECK FILES
# ==================================================

print("\n========================================")
print("       PHASE 8A — LOAD MODELS")
print("========================================")

if not os.path.exists(BASIC_CNN_PATH):
    raise FileNotFoundError(
        f"Basic CNN model not found:\n{BASIC_CNN_PATH}"
    )

if not os.path.exists(CNN3D_PATH):
    raise FileNotFoundError(
        f"3D CNN model not found:\n{CNN3D_PATH}"
    )


# ==================================================
# LOAD BASIC CNN
# ==================================================

print("\nLoading Basic CNN...")

basic_cnn = tf.keras.models.load_model(
    BASIC_CNN_PATH
)

print("Basic CNN loaded successfully.")


# ==================================================
# LOAD 3D CNN
# ==================================================

print("\nLoading 3D CNN...")

cnn3d = tf.keras.models.load_model(
    CNN3D_PATH
)

print("3D CNN loaded successfully.")


# ==================================================
# DISPLAY MODEL INFORMATION
# ==================================================

print("\n========================================")
print("          MODEL INFORMATION")
print("========================================")

print("\nBASIC CNN")
print("Input shape :", basic_cnn.input_shape)
print("Output shape:", basic_cnn.output_shape)
print("Parameters  :", basic_cnn.count_params())

print("\n3D CNN")
print("Input shape :", cnn3d.input_shape)
print("Output shape:", cnn3d.output_shape)
print("Parameters  :", cnn3d.count_params())


# ==================================================
# FINAL STATUS
# ==================================================

print("\n========================================")
print("       PHASE 8A STATUS")
print("========================================")

print("Basic CNN : LOADED")
print("3D CNN    : LOADED")

print("\nBoth trained models loaded successfully.")
print("========================================")