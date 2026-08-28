import streamlit as st


# ==================================================
# DISPLAY PREDICTION
# ==================================================

def display_prediction(label, confidence):
    """
    Display a REAL/FAKE prediction with confidence.
    """

    st.divider()

    if label == "FAKE":

        st.error(
            f"### 🚨 Prediction: {label}"
        )

    else:

        st.success(
            f"### ✅ Prediction: {label}"
        )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.caption(
        "Confidence represents the model's probability-based "
        "prediction for the displayed class."
    )


# ==================================================
# PROCESSING INFORMATION
# ==================================================

def display_processing_info(metadata):
    """
    Display video preprocessing information.
    """

    with st.expander(
        "Video Processing Details"
    ):

        st.write(
            "Sampled frames:",
            metadata.get(
                "sampled_frames",
                "N/A"
            )
        )

        st.write(
            "Faces detected:",
            metadata.get(
                "faces_detected",
                "N/A"
            )
        )

        st.write(
            "Usable face frames:",
            metadata.get(
                "usable_face_frames",
                "N/A"
            )
        )

        st.write(
            "Sequence shape:",
            metadata.get(
                "sequence_shape",
                "N/A"
            )
        )