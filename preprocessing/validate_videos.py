import os
import cv2


# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\original"

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
valid_videos = 0
corrupted_videos = 0

total_frames = 0


# --------------------------------------------------
# VALIDATE VIDEOS
# --------------------------------------------------

print("\n========================================")
print("       VIDEO DATASET VALIDATION")
print("========================================")


for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    video_files = sorted([
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(video_extensions)
    ])

    print(f"\n========== {class_name.upper()} ==========")
    print("Videos found:", len(video_files))

    for filename in video_files:

        total_videos += 1

        video_path = os.path.join(
            class_path,
            filename
        )

        print("\n----------------------------------------")
        print("Class:", class_name)
        print("Video:", filename)

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():

            print("Status: FAILED")
            print("Reason: Could not open video")

            corrupted_videos += 1

            capture.release()
            continue


        # --------------------------------------------------
        # VIDEO PROPERTIES
        # --------------------------------------------------

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )


        # --------------------------------------------------
        # DURATION
        # --------------------------------------------------

        if fps > 0:

            duration = frame_count / fps

        else:

            duration = 0


        # --------------------------------------------------
        # TEST FIRST FRAME
        # --------------------------------------------------

        success, frame = capture.read()


        if (
            not success
            or frame is None
            or frame.size == 0
        ):

            print("Status: FAILED")
            print("Reason: Could not read video frame")

            corrupted_videos += 1

        else:

            print("Status      : VALID")
            print("Frames      :", frame_count)
            print("FPS         :", round(fps, 2))
            print("Resolution  :", f"{width} x {height}")
            print("Duration    :", round(duration, 2), "seconds")

            valid_videos += 1
            total_frames += frame_count


        capture.release()


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("        FINAL VIDEO SUMMARY")
print("========================================")

print("Total videos          :", total_videos)
print("Valid videos          :", valid_videos)
print("Corrupted videos      :", corrupted_videos)
print("Total video frames    :", total_frames)

print("========================================")