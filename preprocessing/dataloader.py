import tensorflow as tf

# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

DATASET_PATH = r"D:\STREESHIELD_Dataset\processed"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# --------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize_images(images, labels):
    """
    Convert pixel values from 0-255 to 0-1.
    """
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    rf"{DATASET_PATH}\train",
    labels="inferred",
    label_mode="binary",
    class_names=["real", "fake"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Apply normalization
train_dataset = train_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    rf"{DATASET_PATH}\valid",
    labels="inferred",
    label_mode="binary",
    class_names=["real", "fake"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

valid_dataset = valid_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# LOAD TEST DATA
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

test_dataset = test_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# PREFETCH DATA
# --------------------------------------------------

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# CHECK NORMALIZATION
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========")
print("Class names:", train_dataset.class_names if hasattr(train_dataset, "class_names") else ["real", "fake"])

for images, labels in train_dataset.take(1):

    print("\n========== FIRST TRAINING BATCH ==========")
    print("Image batch shape :", images.shape)
    print("Label batch shape :", labels.shape)
    print("Image data type   :", images.dtype)

    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())

    print("First labels:", labels[:10].numpy().flatten())

print("\nNormalization successful.")