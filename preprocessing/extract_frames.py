import os
import cv2


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_PATH = r"D:\STREESHIELD_VideoDataset\original"
OUTPUT_PATH = r"D:\STREESHIELD_VideoDataset\frames"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

FRAME_INTERVAL = 20

classes = ["real", "fake"]

video_extensions = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_videos = 0
total_frames_saved = 0
failed_videos = 0


# --------------------------------------------------
# FRAME EXTRACTION
# --------------------------------------------------

print("\n========================================")
print("       VIDEO FRAME EXTRACTION")
print("========================================")


for class_name in classes:

    input_folder = os.path.join(
        INPUT_PATH,
        class_name
    )

    output_folder = os.path.join(
        OUTPUT_PATH,
        class_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    video_files = sorted([
        file
        for file in os.listdir(input_folder)
        if file.lower().endswith(video_extensions)
    ])

    print(f"\n========== {class_name.upper()} ==========")
    print("Videos found:", len(video_files))


    for video_filename in video_files:

        video_path = os.path.join(
            input_folder,
            video_filename
        )

        video_name = os.path.splitext(
            video_filename
        )[0]

        video_output_folder = os.path.join(
            output_folder,
            video_name
        )

        os.makedirs(
            video_output_folder,
            exist_ok=True
        )

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():

            print(
                f"FAILED TO OPEN: "
                f"{class_name}/{video_filename}"
            )

            failed_videos += 1
            continue


        frame_number = 0
        saved_count = 0


        while True:

            success, frame = capture.read()

            if not success:
                break


            if frame_number % FRAME_INTERVAL == 0:

                output_filename = (
                    f"{video_name}_frame_{frame_number:05d}.jpg"
                )

                output_file = os.path.join(
                    video_output_folder,
                    output_filename
                )

                if cv2.imwrite(
                    output_file,
                    frame
                ):

                    saved_count += 1
                    total_frames_saved += 1


            frame_number += 1


        capture.release()

        total_videos += 1

        print(
            f"{class_name}/{video_filename} "
            f"-> {saved_count} frames"
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("       FRAME EXTRACTION SUMMARY")
print("========================================")

print("Videos processed :", total_videos)
print("Frames saved     :", total_frames_saved)
print("Frame interval   :", FRAME_INTERVAL)
print("Failed videos    :", failed_videos)

print("\nFrames saved to:")
print(OUTPUT_PATH)

print("========================================")
                