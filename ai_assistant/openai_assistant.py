import os

from openai import OpenAI


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_MODEL = "gpt-5.4-mini"


# ==================================================
# OPENAI ASSISTANT
# ==================================================

class OpenAISTREESHIELDAssistant:

    def __init__(self, model=None):

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = (
            model
            or os.getenv(
                "OPENAI_MODEL",
                DEFAULT_MODEL
            )
        )

        self.instructions = """
You are the STREESHIELD AI Assistant.

STREESHIELD is a deepfake detection project.

Basic CNN:
- Image-based deepfake detection.
- Input: 128x128 RGB face image.

3D CNN:
- Video-based deepfake detection.
- Input: 16 frames.
- Input shape: (16, 128, 128, 3).

OpenCV:
- Image/video processing.
- Face detection.
- Frame extraction.

Streamlit:
- User interface for image and video detection.

Current baseline results:

Basic CNN:
Accuracy: 67.34%
Precision: 69.44%
Recall: 62.50%
F1-score: 65.79%
ROC-AUC: 74.07%

3D CNN:
Accuracy: 50.00%
Precision: 50.00%
Recall: 100.00%
F1-score: 66.67%
ROC-AUC: 47.00%

The Basic CNN is currently the stronger baseline.

Important:
The 3D CNN predicted all test sequences as FAKE.
Its 100% recall therefore does not mean it is superior.

Do not claim that a specific visual artifact definitely
exists unless evidence is provided.

Model confidence is not a guarantee of correctness.
"""


    def ask(self, question):

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        response = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=question
        )

        return response.output_text


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print("\n========================================")
    print("      STREESHIELD OPENAI TEST")
    print("========================================")

    try:

        assistant = (
            OpenAISTREESHIELDAssistant()
        )

        answer = assistant.ask(
            "How does the 3D CNN work in STREESHIELD?"
        )

        print("\nAssistant:")
        print(answer)

        print("\n========================================")
        print("          12B TEST COMPLETE")
        print("========================================")

    except Exception as error:

        print("\nERROR:")
        print(error)