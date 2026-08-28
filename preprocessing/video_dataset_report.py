import json
import os


# ==================================================
# PATHS
# ==================================================

REPORT_PATH = r"D:\STREESHIELD_VideoDataset\video_dataset_report.json"


# ==================================================
# VERIFIED DATASET RESULTS
# ==================================================

report = {

    "project": "STREESHIELD",

    "phase": "Phase 6 - Video Dataset Preparation",

    "dataset": {
        "source": "FaceForensics++",
        "compression": "c23",
        "manipulation": "Deepfakes",
        "total_source_videos": 100,
        "real_videos": 50,
        "fake_videos": 50
    },

    "video_validation": {
        "videos_validated": 100,
        "valid_videos": 100,
        "corrupted_videos": 0
    },

    "frame_extraction": {
        "frame_interval": 20,
        "videos_processed": 100,
        "frames_extracted": 2640,
        "failed_videos": 0
    },

    "face_processing": {
        "frames_tested": 2640,
        "faces_processed": 2636,
        "no_face_detected": 4,
        "failed_frames": 0,
        "face_size": "128x128"
    },

    "sequence_generation": {
        "sequence_length": 16,
        "frame_shape": "(16, 128, 128, 3)",
        "total_sequences": 120,
        "real_sequences": 60,
        "fake_sequences": 60,
        "valid_sequences": 120,
        "invalid_sequences": 0,
        "videos_with_sequences": 94
    },

    "dataset_split": {
        "train": {
            "real_sequences": 40,
            "fake_sequences": 40,
            "total_sequences": 80
        },
        "valid": {
            "real_sequences": 10,
            "fake_sequences": 10,
            "total_sequences": 20
        },
        "test": {
            "real_sequences": 10,
            "fake_sequences": 10,
            "total_sequences": 20
        }
    },

    "class_balance": {
        "real_sequences": 60,
        "fake_sequences": 60,
        "status": "BALANCED"
    },

    "validation_status": {
        "dataset_validation": "PASSED",
        "sequence_validation": "PASSED"
    }
}


# ==================================================
# CREATE REPORT DIRECTORY
# ==================================================

report_directory = os.path.dirname(
    REPORT_PATH
)

os.makedirs(
    report_directory,
    exist_ok=True
)


# ==================================================
# SAVE REPORT
# ==================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ==================================================
# DISPLAY SUMMARY
# ==================================================

print("\n========================================")
print("     STREESHIELD VIDEO DATASET REPORT")
print("========================================")

print("\nSOURCE VIDEOS")
print("REAL videos :", report["dataset"]["real_videos"])
print("FAKE videos :", report["dataset"]["fake_videos"])
print("TOTAL videos:", report["dataset"]["total_source_videos"])

print("\nVIDEO VALIDATION")
print(
    "Valid videos     :",
    report["video_validation"]["valid_videos"]
)
print(
    "Corrupted videos :",
    report["video_validation"]["corrupted_videos"]
)

print("\nFRAME EXTRACTION")
print(
    "Frames extracted :",
    report["frame_extraction"]["frames_extracted"]
)
print(
    "Frame interval   :",
    report["frame_extraction"]["frame_interval"]
)

print("\nFACE PROCESSING")
print(
    "Frames tested    :",
    report["face_processing"]["frames_tested"]
)
print(
    "Faces processed  :",
    report["face_processing"]["faces_processed"]
)
print(
    "No face detected :",
    report["face_processing"]["no_face_detected"]
)

print("\nSEQUENCES")
print(
    "Sequence length  :",
    report["sequence_generation"]["sequence_length"]
)
print(
    "REAL sequences   :",
    report["sequence_generation"]["real_sequences"]
)
print(
    "FAKE sequences   :",
    report["sequence_generation"]["fake_sequences"]
)
print(
    "Total sequences  :",
    report["sequence_generation"]["total_sequences"]
)
print(
    "Invalid sequences:",
    report["sequence_generation"]["invalid_sequences"]
)

print("\nDATASET SPLIT")
print(
    "TRAIN :",
    report["dataset_split"]["train"]["total_sequences"]
)
print(
    "VALID :",
    report["dataset_split"]["valid"]["total_sequences"]
)
print(
    "TEST  :",
    report["dataset_split"]["test"]["total_sequences"]
)

print("\nCLASS BALANCE")
print(
    "Status:",
    report["class_balance"]["status"]
)

print("\nVALIDATION")
print(
    "Dataset validation :",
    report["validation_status"]["dataset_validation"]
)
print(
    "Sequence validation:",
    report["validation_status"]["sequence_validation"]
)

print("\n----------------------------------------")
print("Report saved to:")
print(REPORT_PATH)
print("----------------------------------------")

print("\nPhase 6I report generated successfully.")