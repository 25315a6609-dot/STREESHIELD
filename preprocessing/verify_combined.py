import os
from PIL import Image

DATASET_PATH = r"D:\STREESHIELD_Dataset"
PROCESSED_PATH = os.path.join(DATASET_PATH, "processed")
RETRY_PATH = os.path.join(DATASET_PATH, "retry_results")

splits = ["train", "valid", "test"]
classes = ["real", "fake"]

extensions = (".jpg", ".jpeg", ".png")

print("\n========================================")
print("       COMBINED DATASET CHECK")
print("========================================\n")

total_combined = 0
total_corrupted = 0

results = {}

for split in splits:

    for class_name in classes:

        processed_folder = os.path.join(
            PROCESSED_PATH, split, class_name
        )

        retry_folder = os.path.join(
            RETRY_PATH, split, class_name
        )

        processed_files = {
            f for f in os.listdir(processed_folder)
            if f.lower().endswith(extensions)
        }

        retry_files = {
            f for f in os.listdir(retry_folder)
            if f.lower().endswith(extensions)
        }

        # Images that would exist after combining both folders
        combined_files = processed_files | retry_files

        corrupted = 0

        for filename in combined_files:

            if filename in retry_files:
                file_path = os.path.join(retry_folder, filename)
            else:
                file_path = os.path.join(processed_folder, filename)

            try:
                with Image.open(file_path) as img:
                    img.verify()

            except Exception:
                corrupted += 1
                total_corrupted += 1
                print(f"Corrupted: {file_path}")

        total_combined += len(combined_files)

        results[f"{split}/{class_name}"] = {
            "processed": len(processed_files),
            "recovered": len(retry_files),
            "combined": len(combined_files),
            "corrupted": corrupted
        }


# --------------------------------------------------
# PRINT COUNTS
# --------------------------------------------------

for split in splits:

    print(split.upper())

    for class_name in classes:

        data = results[f"{split}/{class_name}"]

        print(f"  {class_name.upper()}:")
        print(f"    First processed : {data['processed']}")
        print(f"    Recovered       : {data['recovered']}")
        print(f"    Combined        : {data['combined']}")
        print(f"    Corrupted       : {data['corrupted']}")

    print()


# --------------------------------------------------
# BALANCE
# --------------------------------------------------

print("========== COMBINED CLASS BALANCE ==========\n")

for split in splits:

    real_count = results[f"{split}/real"]["combined"]
    fake_count = results[f"{split}/fake"]["combined"]

    difference = abs(real_count - fake_count)

    print(f"{split.upper()}:")
    print(f"  REAL = {real_count}")
    print(f"  FAKE = {fake_count}")
    print(f"  Difference = {difference}")

    if real_count == fake_count:
        print("  Status = BALANCED")
    else:
        print("  Status = IMBALANCED")

    print()


print("========================================")
print(f"Total combined images : {total_combined}")
print(f"Corrupted images      : {total_corrupted}")
print("========================================")