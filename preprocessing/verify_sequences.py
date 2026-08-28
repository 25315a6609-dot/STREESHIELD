import os
import numpy as np


# --------------------------------------------------
# PATH
# --------------------------------------------------

SEQUENCE_PATH = r"D:\STREESHIELD_VideoDataset\sequences"

classes = ["real", "fake"]

EXPECTED_SHAPE = (16, 128, 128, 3)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

total_sequences = 0
valid_sequences = 0
invalid_sequences = 0

class_counts = {}

videos_with_sequences = 0
videos_without_sequences = 0

videos_without_sequences_list = []


# --------------------------------------------------
# VERIFY
# --------------------------------------------------

print("\n========================================")
print("        SEQUENCE DATASET VERIFICATION")
print("========================================")


for class_name in classes:

    class_path = os.path.join(
        SEQUENCE_PATH,
        class_name
    )

    sequence_count = 0

    video_folders = sorted([
        folder
        for folder in os.listdir(class_path)
        if os.path.isdir(
            os.path.join(class_path, folder)
        )
    ])

    print(f"\n========== {class_name.upper()} ==========")
    print("Video folders:", len(video_folders))


    for video_name in video_folders:

        video_path = os.path.join(
            class_path,
            video_name
        )

        sequence_files = sorted([
            file
            for file in os.listdir(video_path)
            if file.lower().endswith(".npy")
        ])


        if len(sequence_files) == 0:

            videos_without_sequences += 1

            videos_without_sequences_list.append(
                f"{class_name}/{video_name}"
            )

            continue


        videos_with_sequences += 1


        for sequence_file in sequence_files:

            total_sequences += 1
            sequence_count += 1

            sequence_path = os.path.join(
                video_path,
                sequence_file
            )

            try:

                sequence = np.load(
                    sequence_path
                )

                if sequence.shape == EXPECTED_SHAPE:

                    valid_sequences += 1

                else:

                    invalid_sequences += 1

                    print(
                        "INVALID SHAPE:",
                        sequence_path,
                        sequence.shape
                    )

            except Exception as error:

                invalid_sequences += 1

                print(
                    "FAILED TO READ:",
                    sequence_path,
                    error
                )


    class_counts[class_name] = sequence_count

    print(
        "Sequences:",
        sequence_count
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("       FINAL SEQUENCE SUMMARY")
print("========================================")

print("REAL sequences :", class_counts["real"])
print("FAKE sequences :", class_counts["fake"])
print("Total sequences:", total_sequences)

print("\nValid sequences   :", valid_sequences)
print("Invalid sequences :", invalid_sequences)

print(
    "\nVideos with sequences    :",
    videos_with_sequences
)

print(
    "Videos without sequences :",
    videos_without_sequences
)


if videos_without_sequences_list:

    print("\nVideos without 16-frame sequences:")

    for video in videos_without_sequences_list:

        print(" -", video)


print("\nExpected sequence shape:")
print(EXPECTED_SHAPE)

print("========================================")