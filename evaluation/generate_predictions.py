import os
import csv
import tensorflow as tf


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = r"E:\streesheild\models\trained_basic_cnn.keras"
DATASET_PATH = r"D:\STREESHIELD_Dataset\processed\test"
OUTPUT_PATH = r"E:\streesheild\evaluation\test_predictions.csv"


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


# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
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


test_dataset = test_dataset.map(
    normalize_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# GET ACTUAL LABELS
# --------------------------------------------------

actual_labels = []

for images, labels in test_dataset:
    actual_labels.extend(
        labels.numpy().flatten().astype(int)
    )


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()


# --------------------------------------------------
# CONVERT PROBABILITIES TO LABELS
# --------------------------------------------------

predicted_labels = [
    1 if probability >= 0.5 else 0
    for probability in probabilities
]


# --------------------------------------------------
# GET FILE NAMES IN SAME ORDER
# --------------------------------------------------

file_names = []

for class_name in ["real", "fake"]:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = sorted([
        file
        for file in os.listdir(class_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    file_names.extend(
        [
            os.path.join(class_name, file)
            for file in files
        ]
    )


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "actual_label",
        "predicted_probability",
        "predicted_label"
    ])

    for filename, actual, probability, predicted in zip(
        file_names,
        actual_labels,
        probabilities,
        predicted_labels
    ):

        actual_name = "FAKE" if actual == 1 else "REAL"
        predicted_name = "FAKE" if predicted == 1 else "REAL"

        writer.writerow([
            filename,
            actual_name,
            f"{probability:.6f}",
            predicted_name
        ])


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

real_predictions = predicted_labels.count(0)
fake_predictions = predicted_labels.count(1)

print("\n========================================")
print("       PREDICTION SUMMARY")
print("========================================")

print(f"Total predictions : {len(predicted_labels)}")
print(f"Predicted REAL    : {real_predictions}")
print(f"Predicted FAKE    : {fake_predictions}")

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print("========================================")