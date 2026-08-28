import os
import sys


# ==================================================
# ALLOW IMPORT FROM SAME FOLDER
# ==================================================

CURRENT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    CURRENT_FOLDER
)


from video_processor import (
    load_face_detector,
    process_video
)


# ==================================================
# TEST VIDEO
# ==================================================

VIDEO_PATH = (
    r"D:\STREESHIELD_VideoDataset"
    r"\original\real\183.mp4"
)


# ==================================================
# TEST
# ==================================================

print("\n========================================")
print("       VIDEO PROCESSOR TEST")
print("========================================")


if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        f"Test video not found:\n{VIDEO_PATH}"
    )


print(
    "Test video:",
    VIDEO_PATH
)


# --------------------------------------------------
# LOAD DETECTOR
# --------------------------------------------------

detector = load_face_detector()

print(
    "Face detector: loaded successfully"
)


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

sequence, metadata = process_video(
    VIDEO_PATH,
    detector
)


# --------------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------------

print("\nVIDEO INFORMATION")

for key, value in metadata.items():

    print(
        f"{key}: {value}"
    )


# --------------------------------------------------
# DISPLAY SEQUENCE
# --------------------------------------------------

print("\nFINAL SEQUENCE")

print(
    "Shape:",
    sequence.shape
)

print(
    "Dtype:",
    sequence.dtype
)

print(
    "Minimum pixel value:",
    sequence.min()
)

print(
    "Maximum pixel value:",
    sequence.max()
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

expected_shape = (
    16,
    128,
    128,
    3
)

print("\n========================================")

if sequence.shape != expected_shape:

    print(
        "STATUS: VIDEO PROCESSING FAILED"
    )

    print(
        "Expected:",
        expected_shape
    )

    print(
        "Found:",
        sequence.shape
    )

    raise SystemExit(1)


if sequence.min() < 0:

    print(
        "STATUS: FAILED - pixel values below 0"
    )

    raise SystemExit(1)


if sequence.max() > 1:

    print(
        "STATUS: FAILED - pixel values above 1"
    )

    raise SystemExit(1)


print(
    "STATUS: VIDEO PROCESSING PASSED"
)

print("========================================")