import os
import cv2
import numpy as np


# ==================================================
# SETTINGS
# ==================================================

FRAME_INTERVAL = 20
SEQUENCE_LENGTH = 16
IMAGE_SIZE = (128, 128)


# ==================================================
# LOAD FACE DETECTOR
# ==================================================

def load_face_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    if detector.empty():
        raise RuntimeError(
            "Could not load Haar Cascade."
        )

    return detector


# ==================================================
# PROCESS ONE FRAME
# ==================================================

def extract_face(
    frame,
    detector
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    if len(faces) == 0:
        return None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda face:
        face[2] * face[3]
    )

    x, y, w, h = largest_face

    face = frame[
        y:y + h,
        x:x + w
    ]

    face = cv2.resize(
        face,
        IMAGE_SIZE
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = (
        face.astype(np.float32)
        / 255.0
    )

    return face


# ==================================================
# PROCESS VIDEO
# ==================================================

def process_video(
    video_path,
    face_detector
):

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )


    # ==================================================
    # GET VIDEO INFORMATION
    # ==================================================

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():

        raise ValueError(
            "Could not open uploaded video."
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    capture.release()


    # ==================================================
    # FIRST PASS
    # NORMAL INTERVAL = 20
    # ==================================================

    processed_faces = []

    sampled_frames = 0
    faces_found = 0


    capture = cv2.VideoCapture(
        video_path
    )

    frame_number = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        if frame_number % FRAME_INTERVAL == 0:

            sampled_frames += 1

            face = extract_face(
                frame,
                face_detector
            )

            if face is not None:

                faces_found += 1

                processed_faces.append(
                    face
                )

                if len(processed_faces) >= SEQUENCE_LENGTH:
                    break

        frame_number += 1

    capture.release()


    # ==================================================
    # SECOND PASS
    # FALLBACK TO DENSE SAMPLING
    #
    # Only used when interval-20 sampling
    # does not provide 16 face frames.
    # ==================================================

    fallback_used = False
    fallback_sampled_frames = 0
    fallback_faces_found = 0


    if len(processed_faces) < SEQUENCE_LENGTH:

        fallback_used = True

        capture = cv2.VideoCapture(
            video_path
        )

        processed_faces = []

        frame_number = 0


        while True:

            success, frame = capture.read()

            if not success:
                break

            fallback_sampled_frames += 1

            face = extract_face(
                frame,
                face_detector
            )

            if face is not None:

                fallback_faces_found += 1

                processed_faces.append(
                    face
                )

                if len(processed_faces) >= SEQUENCE_LENGTH:
                    break

            frame_number += 1

        capture.release()


    # ==================================================
    # VERIFY
    # ==================================================

    if len(processed_faces) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough face frames for prediction.\n"
            f"Total video frames: {total_frames}\n"
            f"Interval-20 sampled: {sampled_frames}\n"
            f"Interval-20 faces: {faces_found}\n"
            f"Fallback frames scanned: {fallback_sampled_frames}\n"
            f"Fallback faces found: {fallback_faces_found}\n"
            f"Usable face frames: {len(processed_faces)}\n"
            f"Required: {SEQUENCE_LENGTH}"
        )


    # ==================================================
    # CREATE SEQUENCE
    # ==================================================

    sequence = np.array(
        processed_faces[:SEQUENCE_LENGTH],
        dtype=np.float32
    )


    expected_shape = (
        SEQUENCE_LENGTH,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )


    if sequence.shape != expected_shape:

        raise ValueError(
            f"Unexpected sequence shape: "
            f"{sequence.shape}"
        )


    # ==================================================
    # METADATA
    # ==================================================

    metadata = {

        "total_video_frames":
            total_frames,

        "fps":
            fps,

        "sampled_frames":
            sampled_frames,

        "faces_detected":
            faces_found,

        "fallback_used":
            fallback_used,

        "fallback_frames_scanned":
            fallback_sampled_frames,

        "fallback_faces_detected":
            fallback_faces_found,

        "usable_face_frames":
            len(processed_faces),

        "sequence_shape":
            sequence.shape
    }


    return sequence, metadata