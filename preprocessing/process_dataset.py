import os
import cv2

# --------------------------------------------------
# PATHS
# --------------------------------------------------

SOURCE_DATASET = r"D:\STREESHIELD_Dataset"

PROCESSED_DATASET = r"D:\STREESHIELD_Dataset\processed"

# Dataset splits and classes
splits = ["train", "valid", "test"]
classes = ["real", "fake"]

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png")

# --------------------------------------------------
# LOAD HAAR CASCADE
# --------------------------------------------------

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Could not load Haar Cascade.")
    exit()

# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_images = 0
processed_images = 0
no_face_images = 0
failed_images = 0

# Store counts for each folder
results = {}

# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

for split in splits:

    for class_name in classes:

        source_folder = os.path.join(
            SOURCE_DATASET,
            split,
            class_name
        )

        output_folder = os.path.join(
            PROCESSED_DATASET,
            split,
            class_name
        )

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        folder_total = 0
        folder_processed = 0
        folder_no_face = 0
        folder_failed = 0

        print("\n----------------------------------------")
        print(f"Processing: {split}/{class_name}")
        print("----------------------------------------")

        # Get image files
        image_files = [
            file for file in os.listdir(source_folder)
            if file.lower().endswith(image_extensions)
        ]

        for filename in image_files:

            total_images += 1
            folder_total += 1

            input_path = os.path.join(
                source_folder,
                filename
            )

            output_path = os.path.join(
                output_folder,
                filename
            )

            # Read image
            image = cv2.imread(input_path)

            if image is None:
                failed_images += 1
                folder_failed += 1
                print(f"Could not read: {filename}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # Detect faces
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # No face detected
            if len(faces) == 0:

                no_face_images += 1
                folder_no_face += 1

                print(f"No face: {filename}")

                continue

            # Select largest detected face
            largest_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            x, y, w, h = largest_face

            # Crop face
            face = image[
                y:y + h,
                x:x + w
            ]

            # Resize to 128 x 128
            face_resized = cv2.resize(
                face,
                (128, 128)
            )

            # Save processed face
            success = cv2.imwrite(
                output_path,
                face_resized
            )

            if success:

                processed_images += 1
                folder_processed += 1

            else:

                failed_images += 1
                folder_failed += 1

                print(f"Could not save: {filename}")

        # Store results
        results[f"{split}/{class_name}"] = {
            "total": folder_total,
            "processed": folder_processed,
            "no_face": folder_no_face,
            "failed": folder_failed
        }

        print(f"Total images    : {folder_total}")
        print(f"Processed       : {folder_processed}")
        print(f"No face         : {folder_no_face}")
        print(f"Failed          : {folder_failed}")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n\n========================================")
print("       FACE PREPROCESSING SUMMARY")
print("========================================")

print(f"Total images      : {total_images}")
print(f"Processed images  : {processed_images}")
print(f"No face detected  : {no_face_images}")
print(f"Failed images     : {failed_images}")

print("========================================")