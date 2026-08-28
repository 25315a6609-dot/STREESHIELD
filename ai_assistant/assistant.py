"""
STREESHIELD Phase 12A
AI Assistant / Chatbot Module

This module contains:
- Project knowledge
- Question handling
- Conversation history
- Assistant response structure

The actual generative AI model/API will be connected
in Phase 12B.
"""


# ==================================================
# PROJECT KNOWLEDGE
# ==================================================

PROJECT_KNOWLEDGE = {
    "project": "STREESHIELD",

    "purpose": (
        "STREESHIELD is a deepfake detection system that "
        "uses an image-based Basic CNN and a video-based "
        "3D CNN."
    ),

    "basic_cnn": (
        "The Basic CNN analyzes individual 128x128 face "
        "images and predicts REAL or FAKE."
    ),

    "cnn3d": (
        "The 3D CNN analyzes 16-frame video sequences "
        "with input shape (16, 128, 128, 3)."
    ),

    "opencv": (
        "OpenCV is used for image/video processing, "
        "frame extraction, resizing, color conversion, "
        "and face detection."
    ),

    "preprocessing": (
        "Images are resized to 128x128 and normalized "
        "to the 0-1 range. Video inputs are converted "
        "into 16-frame face sequences."
    ),

    "streamlit": (
        "Streamlit provides the user interface for "
        "image and video deepfake detection."
    ),

    "confidence": (
        "Confidence is the model's probability-based "
        "prediction strength. It should not be interpreted "
        "as a guarantee of correctness."
    ),

    "limitations": (
        "The current 3D CNN is a baseline model trained "
        "on a small video-sequence dataset and showed weak "
        "generalization in testing."
    ),

    "best_model": (
        "The Basic CNN is the stronger baseline in the "
        "current experiment, with 67.34% accuracy and "
        "74.07% ROC-AUC."
    )
}


# ==================================================
# ASSISTANT CLASS
# ==================================================

class STREESHIELDAssistant:

    def __init__(self):

        self.project_name = "STREESHIELD"

        self.knowledge = (
            PROJECT_KNOWLEDGE.copy()
        )

        self.history = []


    # ==================================================
    # ADD MESSAGE
    # ==================================================

    def add_message(
        self,
        role,
        message
    ):

        self.history.append(
            {
                "role": role,
                "message": message
            }
        )


    # ==================================================
    # PROJECT KNOWLEDGE
    # ==================================================

    def get_project_knowledge(
        self,
        topic
    ):

        topic = (
            str(topic)
            .lower()
            .strip()
        )

        for key, value in self.knowledge.items():

            if key in topic:

                return value

        return None


    # ==================================================
    # LOCAL RESPONSE
    # ==================================================

    def answer_question(
        self,
        question
    ):
        """
        Temporary local assistant.

        Phase 12B will replace/enhance this with
        an actual generative AI model/API.
        """

        question_clean = (
            str(question)
            .lower()
            .strip()
        )

        self.add_message(
            "user",
            question
        )


        # ----------------------------------------------
        # GREETING
        # ----------------------------------------------

        if question_clean in [
            "hi",
            "hello",
            "hey"
        ]:

            response = (
                "Hello! I am the STREESHIELD assistant. "
                "I can explain the project's CNN, 3D CNN, "
                "preprocessing, confidence scores, and "
                "deepfake detection workflow."
            )


        # ----------------------------------------------
        # WHY FAKE
        # ----------------------------------------------

        elif (
            "why" in question_clean
            and "fake" in question_clean
        ):

            response = (
                "A FAKE prediction means the selected model "
                "found learned patterns that were more "
                "consistent with the FAKE class. A model "
                "prediction does not prove that a specific "
                "visual artifact exists."
            )


        # ----------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------

        elif "confidence" in question_clean:

            response = (
                "Confidence represents how strongly the "
                "model's output favors the predicted class. "
                "A confidence near 50% indicates uncertainty, "
                "while a higher value indicates a stronger "
                "model output. It is not a guarantee that the "
                "prediction is correct."
            )


        # ----------------------------------------------
        # BASIC CNN
        # ----------------------------------------------

        elif (
            "cnn" in question_clean
            and "3d" not in question_clean
        ):

            response = self.knowledge[
                "basic_cnn"
            ]


        # ----------------------------------------------
        # 3D CNN
        # ----------------------------------------------

        elif (
            "3d cnn" in question_clean
            or "video cnn" in question_clean
        ):

            response = self.knowledge[
                "cnn3d"
            ]


        # ----------------------------------------------
        # OPENCV
        # ----------------------------------------------

        elif "opencv" in question_clean:

            response = self.knowledge[
                "opencv"
            ]


        # ----------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------

        elif (
            "preprocess" in question_clean
            or "preprocessing" in question_clean
        ):

            response = self.knowledge[
                "preprocessing"
            ]


        # ----------------------------------------------
        # STREAMLIT
        # ----------------------------------------------

        elif "streamlit" in question_clean:

            response = self.knowledge[
                "streamlit"
            ]


        # ----------------------------------------------
        # BEST MODEL
        # ----------------------------------------------

        elif (
            "best model" in question_clean
            or "which model" in question_clean
        ):

            response = self.knowledge[
                "best_model"
            ]


        # ----------------------------------------------
        # LIMITATIONS
        # ----------------------------------------------

        elif "limitation" in question_clean:

            response = self.knowledge[
                "limitations"
            ]


        # ----------------------------------------------
        # DEEPFAKE
        # ----------------------------------------------

        elif "deepfake" in question_clean:

            response = (
                "A deepfake is synthetic or manipulated "
                "media designed to imitate a real person "
                "or event. STREESHIELD attempts to classify "
                "uploaded face images and video sequences "
                "as REAL or FAKE."
            )


        # ----------------------------------------------
        # UNKNOWN QUESTION
        # ----------------------------------------------

        else:

            response = (
                "I can answer questions about STREESHIELD, "
                "deepfake detection, Basic CNN, 3D CNN, "
                "OpenCV, preprocessing, Streamlit, "
                "confidence scores, limitations, and "
                "the project's methodology."
            )


        self.add_message(
            "assistant",
            response
        )

        return response


    # ==================================================
    # CLEAR HISTORY
    # ==================================================

    def clear_history(self):

        self.history = []


# ==================================================
# 12A TEST
# ==================================================

if __name__ == "__main__":

    assistant = STREESHIELDAssistant()


    print("\n========================================")
    print("       STREESHIELD AI ASSISTANT")
    print("========================================")


    questions = [
        "Hello",
        "How does CNN work?",
        "What does confidence mean?",
        "How does 3D CNN work?",
        "Why is this fake?"
    ]


    for question in questions:

        print(
            f"\nUser: {question}"
        )

        answer = assistant.answer_question(
            question
        )

        print(
            f"Assistant: {answer}"
        )


    print("\n========================================")
    print("          12A TEST COMPLETE")
    print("========================================")