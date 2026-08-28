import tensorflow as tf


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = r"E:\streesheild\models\trained_basic_cnn.keras"
DATASET_PATH = r"D:\STREESHIELD_Dataset\processed"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Trained CNN model loaded successfully.")
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    rf"{DATASET_PATH}\test",
    labels="inferred",
    label_mode="binary",
    class_names=["real", "fake"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# NORMALIZE TEST IMAGES
# --------------------------------------------------

def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


test_dataset = test_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

print("\n========================================")
print("         TEST DATASET EVALUATION")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========================================")
print("          TEST RESULTS")
print("========================================")

print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")

print("========================================")