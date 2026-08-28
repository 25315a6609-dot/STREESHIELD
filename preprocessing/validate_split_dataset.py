import os
import numpy as np


# ==================================================
# PATHS
# ==================================================

DATASET_PATH = r"D:\STREESHIELD_VideoDataset\split_sequences"


# ==================================================
# SETTINGS
# ==================================================

SPLITS = ["train", "valid", "test"]
CLASSES = ["real", "fake"]

EXPECTED_SHAPE = (16, 128, 128, 3)


# ==================================================
# GLOBAL COUNTERS
# ==================================================

grand_total_sequences = 0
grand_valid_sequences = 0
grand_invalid_sequences = 0

grand_total_videos = 0
grand_videos_with_sequences = 0

class_sequence_totals = {
    "real": 0,
    "fake": 0
}


# ==================================================
# VALIDATE DATASET
# ==================================================

print("\n========================================")
print("     FINAL VIDEO DATASET VALIDATION")
print("========================================")


for split in SPLITS:

    print(f"\n========== {split.upper()} ==========")

    split_total_sequences = 0
    split_valid_sequences = 0
    split_invalid_sequences = 0

    split_video_count = 0

    for class_name in CLASSES:

        class_path = os.path.join(
            DATASET_PATH,
            split,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                f"ERROR: Missing folder: {class_path}"
            )

            continue


        video_folders = sorted([
            folder
            for folder in os.listdir(class_path)
            if os.path.isdir(
                os.path.join(
                    class_path,
                    folder
                )
            )
        ])


        class_sequences = 0
        class_valid = 0
        class_invalid = 0


        for video_name in video_folders:

            split_video_count += 1
            grand_total_videos += 1

            video_path = os.path.join(
                class_path,
                video_name
            )

            sequence_files = [
                file
                for file in os.listdir(video_path)
                if file.lower().endswith(".npy")
            ]


            if len(sequence_files) > 0:

                grand_videos_with_sequences += 1


            for sequence_file in sequence_files:

                split_total_sequences += 1
                class_sequences += 1

                grand_total_sequences += 1
                class_sequence_totals[class_name] += 1


                sequence_path = os.path.join(
                    video_path,
                    sequence_file
                )


                try:

                    sequence = np.load(
                        sequence_path,
                        allow_pickle=False
                    )


                    if sequence.shape == EXPECTED_SHAPE:

                        split_valid_sequences += 1
                        class_valid += 1

                        grand_valid_sequences += 1

                    else:

                        split_invalid_sequences += 1
                        class_invalid += 1

                        grand_invalid_sequences += 1

                        print(
                            "\nINVALID SHAPE"
                        )

                        print(
                            "File:",
                            sequence_path
                        )

                        print(
                            "Expected:",
                            EXPECTED_SHAPE
                        )

                        print(
                            "Found:",
                            sequence.shape
                        )


                except Exception as error:

                    split_invalid_sequences += 1
                    class_invalid += 1

                    grand_invalid_sequences += 1

                    print(
                        "\nFAILED TO READ"
                    )

                    print(
                        "File:",
                        sequence_path
                    )

                    print(
                        "Error:",
                        error
                    )


        print(
            f"{class_name.upper()} "
            f"videos={len(video_folders)}, "
            f"sequences={class_sequences}, "
            f"valid={class_valid}, "
            f"invalid={class_invalid}"
        )


    print("\nSplit totals:")

    print(
        "Videos    :",
        split_video_count
    )

    print(
        "Sequences :",
        split_total_sequences
    )

    print(
        "Valid     :",
        split_valid_sequences
    )

    print(
        "Invalid   :",
        split_invalid_sequences
    )


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========================================")
print("       FINAL VALIDATION SUMMARY")
print("========================================")

print(
    "Total videos processed :",
    grand_total_videos
)

print(
    "Videos with sequences  :",
    grand_videos_with_sequences
)

print(
    "REAL sequences         :",
    class_sequence_totals["real"]
)

print(
    "FAKE sequences         :",
    class_sequence_totals["fake"]
)

print(
    "Total sequences        :",
    grand_total_sequences
)

print(
    "Valid sequences        :",
    grand_valid_sequences
)

print(
    "Invalid sequences      :",
    grand_invalid_sequences
)

print(
    "Expected shape         :",
    EXPECTED_SHAPE
)


# ==================================================
# BALANCE CHECK
# ==================================================

real_count = class_sequence_totals["real"]
fake_count = class_sequence_totals["fake"]

print("\n========== CLASS BALANCE ==========")

print("REAL:", real_count)
print("FAKE:", fake_count)

if real_count == fake_count:

    print("Status: BALANCED")

else:

    difference = abs(
        real_count - fake_count
    )

    print("Status: IMBALANCED")
    print("Difference:", difference)


# ==================================================
# FINAL STATUS
# ==================================================

print("\n========================================")

if (
    grand_total_sequences == 120
    and grand_valid_sequences == 120
    and grand_invalid_sequences == 0
    and real_count == 60
    and fake_count == 60
):

    print(
        "STATUS: DATASET VALIDATION PASSED"
    )

else:

    print(
        "STATUS: DATASET VALIDATION NEEDS ATTENTION"
    )

print("========================================")