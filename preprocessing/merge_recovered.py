import os
import shutil

DATASET_PATH = r"D:\STREESHIELD_Dataset"

RETRY_PATH = os.path.join(DATASET_PATH, "retry_results")
PROCESSED_PATH = os.path.join(DATASET_PATH, "processed")

splits = ["train", "valid", "test"]
classes = ["real", "fake"]

copied = 0


for split in splits:

    for class_name in classes:

        retry_folder = os.path.join(
            RETRY_PATH,
            split,
            class_name
        )

        processed_folder = os.path.join(
            PROCESSED_PATH,
            split,
            class_name
        )

        if not os.path.exists(retry_folder):
            continue

        os.makedirs(processed_folder, exist_ok=True)

        for filename in os.listdir(retry_folder):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            source = os.path.join(
                retry_folder,
                filename
            )

            destination = os.path.join(
                processed_folder,
                filename
            )

            # Safety check: don't overwrite an existing file
            if os.path.exists(destination):
                print(f"Already exists, skipped: {filename}")
                continue

            shutil.copy2(source, destination)

            copied += 1

            print(
                f"Copied: {split}/{class_name}/{filename}"
            )


print("\n========================================")
print("        MERGE COMPLETE")
print("========================================")
print(f"Recovered images copied : {copied}")
print("========================================")