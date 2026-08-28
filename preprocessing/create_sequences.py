import os
import cv2
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_PATH = r"D:\STREESHIELD_VideoDataset\processed_frames"
OUTPUT_PATH = r"D:\STREESHIELD_VideoDataset\sequences"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SEQUENCE_LENGTH = 16
IMAGE_SIZE = (128, 128)

classes = ["real", "fake"]

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_videos = 0
videos_with_sequences = 0
videos_too_short = 0
total_sequences = 0


# --------------------------------------------------
# CREATE SEQUENCES
# --------------------------------------------------

print("\n========================================")
print("       16-FRAME SEQUENCE CREATION")
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

        total_videos += 1

        video_folder = os.path.join(
            class_input_path,
            video_name
        )

        frame_files = sorted([
            file
            for file in os.listdir(video_folder)
            if file.lower().endswith(
                image_extensions
            )
        ])


        print(
            f"\n{class_name}/{video_name}"
        )

        print(
            "Available processed frames:",
            len(frame_files)
        )


        # --------------------------------------------------
        # CHECK IF ENOUGH FRAMES EXIST
        # --------------------------------------------------

        if len(frame_files) < SEQUENCE_LENGTH:

            print(
                "Status: TOO SHORT"
            )

            videos_too_short += 1

            continue


        # --------------------------------------------------
        # OUTPUT FOLDER
        # --------------------------------------------------

        video_output_folder = os.path.join(
            class_output_path,
            video_name
        )

        os.makedirs(
            video_output_folder,
            exist_ok=True
        )


        # --------------------------------------------------
        # CREATE SEQUENCES
        # --------------------------------------------------

        sequence_number = 0


        for start in range(
            0,
            len(frame_files) - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH
        ):

            selected_frames = frame_files[
                start:start + SEQUENCE_LENGTH
            ]


            sequence = []

            valid_sequence = True


            # --------------------------------------------------
            # LOAD EACH FRAME
            # --------------------------------------------------

            for frame_file in selected_frames:

                frame_path = os.path.join(
                    video_folder,
                    frame_file
                )

                frame = cv2.imread(
                    frame_path
                )


                if frame is None:

                    valid_sequence = False
                    break


                frame = cv2.resize(
                    frame,
                    IMAGE_SIZE
                )


                # OpenCV BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                # Normalize pixel values
                frame = frame.astype(
                    np.float32
                ) / 255.0


                sequence.append(
                    frame
                )


            # --------------------------------------------------
            # SKIP INVALID SEQUENCE
            # --------------------------------------------------

            if not valid_sequence:
                continue


            # --------------------------------------------------
            # CONVERT TO NUMPY ARRAY
            # --------------------------------------------------

            sequence = np.array(
                sequence,
                dtype=np.float32
            )


            # Expected:
            # (16, 128, 128, 3)

            output_file = os.path.join(
                video_output_folder,
                f"{video_name}_sequence_{sequence_number:03d}.npy"
            )


            np.save(
                output_file,
                sequence
            )


            sequence_number += 1
            total_sequences += 1


        if sequence_number > 0:

            videos_with_sequences += 1


        print(
            "Sequences created:",
            sequence_number
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("     SEQUENCE CREATION SUMMARY")
print("========================================")

print("Total videos processed :", total_videos)
print("Videos with sequences  :", videos_with_sequences)
print("Videos too short       :", videos_too_short)
print("Total sequences        :", total_sequences)
print("Sequence length        :", SEQUENCE_LENGTH)
print("Frame size             :", IMAGE_SIZE)

print("\nSequences saved to:")
print(OUTPUT_PATH)

print("========================================")