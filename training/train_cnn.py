import json
import tensorflow as tf


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = r"E:\streesheild\models\basic_cnn.keras"
DATASET_PATH = r"D:\STREESHIELD_Dataset\processed"
HISTORY_PATH = r"E:\streesheild\training\training_history.json"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10


# --------------------------------------------------
# LOAD CNN MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("CNN model loaded successfully.")
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# --------------------------------------------------
# COMPILE CNN
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("CNN compiled successfully.")


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


# --------------------------------------------------
# NORMALIZE IMAGES
# --------------------------------------------------

def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_dataset = train_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_dataset = valid_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
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
# VERIFY DATASET
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("Class names  : ['real', 'fake']")
print("Batch size   :", BATCH_SIZE)
print("Image size   :", IMAGE_SIZE)

for images, labels in train_dataset.take(1):

    print("\n========== FIRST TRAINING BATCH ==========")
    print("Image shape  :", images.shape)
    print("Label shape  :", labels.shape)
    print("Data type    :", images.dtype)
    print("Min pixel    :", tf.reduce_min(images).numpy())
    print("Max pixel    :", tf.reduce_max(images).numpy())


# --------------------------------------------------
# TRAIN CNN
# --------------------------------------------------

print("\n========================================")
print("          STARTING CNN TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# --------------------------------------------------
# TRAINING RESULTS
# --------------------------------------------------

print("\n========================================")
print("          CNN TRAINING COMPLETE")
print("========================================")

print("\nFinal training accuracy:",
      history.history["accuracy"][-1])

print("Final validation accuracy:",
      history.history["val_accuracy"][-1])

print("Final training loss:",
      history.history["loss"][-1])

print("Final validation loss:",
      history.history["val_loss"][-1])


# --------------------------------------------------
# SAVE TRAINING HISTORY
# --------------------------------------------------

with open(HISTORY_PATH, "w") as file:
    json.dump(history.history, file, indent=4)

print("\nTraining history saved successfully.")
print("Saved to:", HISTORY_PATH)
# --------------------------------------------------
# SAVE TRAINED CNN MODEL
# --------------------------------------------------

TRAINED_MODEL_PATH = r"E:\streesheild\models\trained_basic_cnn.keras"

model.save(TRAINED_MODEL_PATH)

print("\nTrained CNN model saved successfully.")
print("Saved to:", TRAINED_MODEL_PATH)