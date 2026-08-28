
import os
from collections import Counter
from PIL import Image

# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

ORIGINAL_DATASET = r"D:\STREESHIELD_Dataset"
PROCESSED_DATASET = r"D:\STREESHIELD_Dataset\processed"

splits = ["train", "valid", "test"]
classes = ["real", "fake"]

image_extensions = (".jpg", ".jpeg", ".png")


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_original = 0
total_processed = 0
total_corrupted = 0

format_counts = Counter()
dimension_counts = Counter()

results = {}


# --------------------------------------------------
# VERIFY EACH FOLDER
# --------------------------------------------------

for split in splits:

    for class_name in classes:

        original_folder = os.path.join(
            ORIGINAL_DATASET,
            split,
            class_name
        )

        processed_folder = os.path.join(
            PROCESSED_DATASET,
            split,
            class_name
        )

        # Original images
        original_files = [
            file for file in os.listdir(original_folder)
            if file.lower().endswith(image_extensions)
        ]

        # Processed images
        processed_files = [
            file for file in os.listdir(processed_folder)
            if file.lower().endswith(image_extensions)
        ]

        original_count = len(original_files)
        processed_count = len(processed_files)

        no_face_count = original_count - processed_count

        total_original += original_count
        total_processed += processed_count

        corrupted_count = 0

        # --------------------------------------------------
        # CHECK PROCESSED IMAGES
        # --------------------------------------------------

        for filename in processed_files:

            file_path = os.path.join(
                processed_folder,
                filename
            )

            try:

                with Image.open(file_path) as img:

                    img.verify()

                with Image.open(file_path) as img:

                    width, height = img.size
                    image_format = img.format

                format_counts[image_format] += 1
                dimension_counts[(width, height)] += 1

            except Exception:

                corrupted_count += 1
                total_corrupted += 1

                print(f"Corrupted processed image: {file_path}")

        results[f"{split}/{class_name}"] = {
            "original": original_count,
            "processed": processed_count,
            "no_face": no_face_count,
            "corrupted": corrupted_count
        }


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n========================================")
print("      PROCESSED DATASET VERIFICATION")
print("========================================\n")


for split in splits:

    print(f"{split.upper()}")

    for class_name in classes:

        key = f"{split}/{class_name}"

        data = results[key]

        print(f"  {class_name.upper()}:")
        print(f"    Original  : {data['original']}")
        print(f"    Processed : {data['processed']}")
        print(f"    No face   : {data['no_face']}")
        print(f"    Corrupted : {data['corrupted']}")

    print()


# --------------------------------------------------
# CLASS BALANCE
# --------------------------------------------------

print("========== PROCESSED CLASS BALANCE ==========\n")

for split in splits:

    real_count = results[f"{split}/real"]["processed"]
    fake_count = results[f"{split}/fake"]["processed"]

    print(f"{split.upper()}:")
    print(f"  REAL = {real_count}")
    print(f"  FAKE = {fake_count}")

    if real_count == fake_count:
        print("  Status = BALANCED")
    else:
        print("  Status = IMBALANCED")

    print()


# --------------------------------------------------
# IMAGE DIMENSIONS
# --------------------------------------------------

print("========== PROCESSED IMAGE DIMENSIONS ==========\n")

for dimensions, count in dimension_counts.most_common():

    width, height = dimensions

    print(f"{width} x {height} : {count}")


# --------------------------------------------------
# IMAGE FORMATS
# --------------------------------------------------

print("\n========== PROCESSED IMAGE FORMATS ==========\n")

for image_format, count in format_counts.items():

    print(f"{image_format} : {count}")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("             FINAL SUMMARY")
print("========================================")

print(f"Original images     : {total_original}")
print(f"Processed images    : {total_processed}")
print(f"No-face images      : {total_original - total_processed}")
print(f"Corrupted processed : {total_corrupted}")

print("========================================")