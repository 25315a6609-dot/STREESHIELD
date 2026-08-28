
import os
import shutil

DATASET_PATH = r"D:\STREESHIELD_Dataset"

splits = ["train", "valid", "test"]
classes = ["real", "fake"]

# Where sample no-face images will be copied
OUTPUT_PATH = r"D:\STREESHIELD_Dataset\no_face_samples"

# Number of samples to inspect from each split/class
SAMPLE_COUNT = 5


for split in splits:

    for class_name in classes:

        original_folder = os.path.join(
            DATASET_PATH,
            split,
            class_name
        )

        processed_folder = os.path.join(
            DATASET_PATH,
            "processed",
            split,
            class_name
        )

        output_folder = os.path.join(
            OUTPUT_PATH,
            split,
            class_name
        )

        os.makedirs(output_folder, exist_ok=True)

        original_files = {
            file
            for file in os.listdir(original_folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        processed_files = {
            file
            for file in os.listdir(processed_folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        # Images that were not processed = no face detected
        no_face_files = sorted(original_files - processed_files)

        print(f"\n{split}/{class_name}")
        print(f"No-face images found: {len(no_face_files)}")

        # Copy a small sample
        samples = no_face_files[:SAMPLE_COUNT]

        for filename in samples:

            source = os.path.join(
                original_folder,
                filename
            )

            destination = os.path.join(
                output_folder,
                filename
            )

            shutil.copy2(source, destination)

            print(f"Sample copied: {filename}")


print("\n========================================")
print("No-face sample extraction complete.")
print(f"Samples saved in: {OUTPUT_PATH}")
print("========================================")