"""
STREESHIELD AI ASSISTANT
Phase 13D

Clean Streamlit-native chat interface.

This file intentionally does NOT use:
- HTML <div> elements
- SVG links
- raw HTML rendering
- unsafe_allow_html=True

The interface uses only Streamlit components so that
HTML source code can never appear on the screen.
"""

import streamlit as st


# ============================================================
# ASSISTANT KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = {

    "streeshield": (
        "STREESHIELD is an AI-powered deepfake detection system "
        "that analyzes images and videos using deep learning, "
        "computer vision, AI analysis and risk assessment."
    ),

    "basic cnn": (
        "The Basic CNN is the image-based deepfake detection model. "
        "It receives a processed 128x128 RGB face image and produces "
        "a binary prediction for REAL or FAKE."
    ),

    "3d cnn": (
        "The 3D CNN is the video-based model. It analyzes a sequence "
        "of 16 face frames with shape 16x128x128x3 and is designed "
        "to learn both spatial and temporal information."
    ),

    "face detection": (
        "STREESHIELD uses OpenCV face detection before image analysis. "
        "For an image, the largest detected face is cropped, resized "
        "to 128x128 pixels and normalized before being sent to the CNN."
    ),

    "preprocessing": (
        "Image preprocessing converts the image to RGB, resizes the "
        "face to 128x128 pixels and normalizes pixel values to the "
        "range 0 to 1."
    ),

    "confidence": (
        "Confidence indicates how strongly the model supports its "
        "prediction. A confidence close to 50 percent indicates "
        "greater uncertainty."
    ),

    "real": (
        "REAL means the model classified the uploaded media as "
        "more likely to be authentic according to its learned patterns."
    ),

    "fake": (
        "FAKE means the model classified the uploaded media as "
        "more likely to contain manipulated or synthetic content."
    ),

    "opencv": (
        "OpenCV is used for computer-vision operations such as "
        "reading images and videos, converting color spaces, "
        "detecting faces and preparing image data."
    ),

    "risk": (
        "Risk assessment converts the model prediction and confidence "
        "into a risk category such as LOW, MEDIUM or HIGH."
    ),

    "report": (
        "The AI report summarizes the detection type, model used, "
        "prediction, confidence and AI explanation."
    ),

    "video": (
        "In video mode, STREESHIELD processes the uploaded video, "
        "detects faces, creates a 16-frame sequence and sends the "
        "sequence to the 3D CNN."
    ),

    "image": (
        "In image mode, STREESHIELD detects the largest face, "
        "preprocesses it into a 128x128 RGB image and sends it "
        "to the Basic CNN."
    ),

    "limitations": (
        "Deepfake detection is not perfect. Image quality, lighting, "
        "compression, unusual facial characteristics and differences "
        "between training and real-world data can affect predictions."
    ),

    "methodology": (
        "The STREESHIELD workflow is: upload media, preprocess the "
        "input, run the appropriate CNN model, obtain REAL/FAKE "
        "prediction and confidence, perform AI analysis, calculate "
        "risk and generate a report."
    ),

    "cnn": (
        "CNN stands for Convolutional Neural Network. It is commonly "
        "used for image analysis because convolutional layers can "
        "learn visual patterns such as edges, textures and facial "
        "features."
    ),

    "temporal": (
        "Temporal analysis means analyzing information across multiple "
        "video frames. The 3D CNN uses a 16-frame sequence to learn "
        "spatial and temporal patterns."
    ),
}


# ============================================================
# FIND ANSWER
# ============================================================

def get_assistant_response(question):
    """
    Generate a simple project-specific response.

    The function intentionally returns plain text only.
    No HTML or SVG is generated.
    """

    question = question.strip().lower()

    if not question:
        return (
            "Please enter a question about STREESHIELD, "
            "deepfake detection, CNN, 3D CNN, preprocessing "
            "or the project methodology."
        )

    # Direct keyword matching
    for keyword, response in KNOWLEDGE_BASE.items():

        if keyword in question:

            return response

    # Additional question patterns

    if "how does" in question and "work" in question:

        return (
            "STREESHIELD works by accepting an image or video, "
            "preprocessing the input, running the appropriate "
            "deep learning model, calculating confidence, "
            "performing AI analysis and generating a risk assessment "
            "and report."
        )

    if "which model" in question:

        return (
            "STREESHIELD uses two main models. Basic CNN is used "
            "for image detection and 3D CNN is used for video "
            "sequence detection."
        )

    if "difference" in question and "cnn" in question:

        return (
            "The Basic CNN analyzes a single 128x128 face image. "
            "The 3D CNN analyzes 16 consecutive 128x128 face frames "
            "and can learn temporal information from the sequence."
        )

    if "why" in question and "face" in question:

        return (
            "Face preprocessing helps the model focus on the facial "
            "region instead of unrelated background information."
        )

    if "phase 13d" in question:

        return (
            "Phase 13D integrates the existing detection pipeline "
            "with AI analysis, risk assessment, AI reports and the "
            "STREESHIELD AI Assistant."
        )

    return (
        "I can help with STREESHIELD, deepfake detection, "
        "Basic CNN, 3D CNN, face detection, preprocessing, "
        "confidence scores, OpenCV, risk assessment, reports "
        "and project methodology."
    )


# ============================================================
# ASSISTANT HEADER
# ============================================================

def render_chat():

    st.markdown("## 🤖 STREESHIELD AI Assistant")

    st.write(
        "Ask questions about deepfake detection, CNN, 3D CNN, "
        "confidence scores, preprocessing, OpenCV, or the "
        "STREESHIELD project."
    )

    st.divider()

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "streeshield_chat_messages" not in st.session_state:

        st.session_state.streeshield_chat_messages = [

            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I am the STREESHIELD AI Assistant. "
                    "Ask me about deepfake detection, Basic CNN, "
                    "3D CNN, preprocessing, confidence scores, "
                    "risk assessment or the project methodology."
                )
            }

        ]

    # ========================================================
    # DISPLAY CHAT HISTORY
    # ========================================================

    for message in st.session_state.streeshield_chat_messages:

        with st.chat_message(message["role"]):

            st.write(
                message["content"]
            )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask STREESHIELD AI Assistant..."
    )

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.streeshield_chat_messages.append(

            {
                "role": "user",
                "content": question
            }

        )

        with st.chat_message("user"):

            st.write(question)

        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        response = get_assistant_response(
            question
        )

        st.session_state.streeshield_chat_messages.append(

            {
                "role": "assistant",
                "content": response
            }

        )

        with st.chat_message("assistant"):

            st.write(response)

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Assistant Chat",
        key="clear_streeshield_chat"
    ):

        st.session_state.streeshield_chat_messages = [

            {
                "role": "assistant",
                "content": (
                    "Chat cleared. 👋 "
                    "Ask me a new question about STREESHIELD."
                )
            }

        ]

        st.rerun()