import os
import cv2

SOURCE_DATASET = r"D:\STREESHIELD_Dataset"
PROCESSED_DATASET = r"D:\STREESHIELD_Dataset\processed"
RETRY_OUTPUT = r"D:\STREESHIELD_Dataset\retry_results"

splits = ["train", "valid", "test"]
classes = ["real", "fake"]
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

total_no_face = 0
recovered = 0
still_missing = 0
failed = 0


# --------------------------------------------------
# PROCESS ALL NO-FACE IMAGES
# --------------------------------------------------

for split in splits:

    for class_name in classes:

        original_folder = os.path.join(
            SOURCE_DATASET,
            split,
            class_name
        )

        processed_folder = os.path.join(
            PROCESSED_DATASET,
            split,
            class_name
        )

        output_folder = os.path.join(
            RETRY_OUTPUT,
            split,
            class_name
        )

        os.makedirs(output_folder, exist_ok=True)

        original_files = {
            file
            for file in os.listdir(original_folder)
            if file.lower().endswith(image_extensions)
        }

        processed_files = {
            file
            for file in os.listdir(processed_folder)
            if file.lower().endswith(image_extensions)
        }

        # These are the images missed by the first detector
        no_face_files = sorted(original_files - processed_files)

        print(f"\nProcessing {split}/{class_name}")
        print(f"No-face images: {len(no_face_files)}")

        for filename in no_face_files:

            total_no_face += 1

            input_path = os.path.join(
                original_folder,
                filename
            )

            image = cv2.imread(input_path)

            if image is None:
                failed += 1
                print(f"Could not read: {filename}")
                continue

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # More tolerant Haar Cascade settings
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

            if len(faces) == 0:

                still_missing += 1
                print(f"Still no face: {filename}")

                continue

            # Select the largest detected face
            largest_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            x, y, w, h = largest_face

            face = image[
                y:y + h,
                x:x + w
            ]

            face_resized = cv2.resize(
                face,
                (128, 128)
            )

            output_path = os.path.join(
                output_folder,
                filename
            )

            if cv2.imwrite(output_path, face_resized):

                recovered += 1

                print(f"Recovered: {filename}")

            else:

                failed += 1
                print(f"Could not save: {filename}")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n========================================")
print("     FULL NO-FACE RETRY SUMMARY")
print("========================================")

print(f"Images tested       : {total_no_face}")
print(f"Faces recovered     : {recovered}")
print(f"Still not detected  : {still_missing}")
print(f"Failed              : {failed}")

print("========================================")