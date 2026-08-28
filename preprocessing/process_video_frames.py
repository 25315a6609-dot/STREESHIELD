import os
import cv2


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_PATH = r"D:\STREESHIELD_VideoDataset\frames"
OUTPUT_PATH = r"D:\STREESHIELD_VideoDataset\processed_frames"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (128, 128)

classes = ["real", "fake"]

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


# --------------------------------------------------
# LOAD HAAR CASCADE
# --------------------------------------------------

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    cascade_path
)

if face_detector.empty():

    print("ERROR: Could not load Haar Cascade.")
    exit()


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_frames = 0
faces_detected = 0
no_face = 0
failed = 0


# --------------------------------------------------
# PROCESS FRAMES
# --------------------------------------------------

print("\n========================================")
print("     VIDEO FRAME FACE PROCESSING")
print("========================================")


for class_name in classes:

    class_input_path = os.path.join(
        INPUT_PATH,
        class_name
    )

    class_output_path = os.path.join(
        OUTPUT_PATH,
        class_name
    )

    os.makedirs(
        class_output_path,
        exist_ok=True
    )


    video_folders = sorted([
        folder
        for folder in os.listdir(class_input_path)
        if os.path.isdir(
            os.path.join(
                class_input_path,
                folder
            )
        )
    ])


    print(f"\n========== {class_name.upper()} ==========")
    print("Videos found:", len(video_folders))


    for video_name in video_folders:

        input_folder = os.path.join(
            class_input_path,
            video_name
        )

        output_folder = os.path.join(
            class_output_path,
            video_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )


        frame_files = sorted([
            file
            for file in os.listdir(input_folder)
            if file.lower().endswith(
                image_extensions
            )
        ])


        video_total = 0
        video_faces = 0
        video_no_face = 0


        for filename in frame_files:

            total_frames += 1
            video_total += 1


            input_file = os.path.join(
                input_folder,
                filename
            )

            output_file = os.path.join(
                output_folder,
                filename
            )


            # --------------------------------------------------
            # READ FRAME
            # --------------------------------------------------

            image = cv2.imread(
                input_file
            )

            if image is None:

                failed += 1

                continue


            # --------------------------------------------------
            # GRAYSCALE
            # --------------------------------------------------

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )


            # --------------------------------------------------
            # FACE DETECTION
            # --------------------------------------------------

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )


            # --------------------------------------------------
            # NO FACE
            # --------------------------------------------------

            if len(faces) == 0:

                no_face += 1
                video_no_face += 1

                continue


            # --------------------------------------------------
            # SELECT LARGEST FACE
            # --------------------------------------------------

            largest_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            x, y, w, h = largest_face


            # --------------------------------------------------
            # CROP FACE
            # --------------------------------------------------

            face = image[
                y:y + h,
                x:x + w
            ]


            # --------------------------------------------------
            # RESIZE
            # --------------------------------------------------

            face_resized = cv2.resize(
                face,
                IMAGE_SIZE
            )


            # --------------------------------------------------
            # SAVE FACE
            # --------------------------------------------------

            success = cv2.imwrite(
                output_file,
                face_resized
            )

            if success:

                faces_detected += 1
                video_faces += 1

            else:

                failed += 1


        print(
            f"{class_name}/{video_name} "
            f"-> Frames: {video_total}, "
            f"Faces: {video_faces}, "
            f"No face: {video_no_face}"
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("      VIDEO FACE PROCESSING SUMMARY")
print("========================================")

print("Total frames tested :", total_frames)
print("Faces processed     :", faces_detected)
print("No face detected    :", no_face)
print("Failed              :", failed)

print("\nProcessed faces saved to:")
print(OUTPUT_PATH)

print("========================================")