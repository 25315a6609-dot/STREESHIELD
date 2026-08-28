"""
STREESHIELD Phase 12D
Project Knowledge Base

Central knowledge source for the AI Assistant.
"""

STREESHIELD_KNOWLEDGE = """

# ==================================================
# STREESHIELD
# ==================================================

STREESHIELD is an AI-assisted deepfake detection
system that analyzes images and videos and provides
prediction, confidence, AI analysis, risk assessment,
reporting, and conversational assistance.


# ==================================================
# IMAGE DETECTION
# ==================================================

The image detection pipeline uses the Basic CNN.

Input:
128 x 128 RGB face image.

Processing:
1. Image is read using OpenCV.
2. A face is detected when a full-size image is uploaded.
3. The largest detected face is selected.
4. The face is resized to 128 x 128.
5. BGR is converted to RGB.
6. Pixel values are normalized to the 0-1 range.
7. The Basic CNN produces a probability.
8. Probability >= 0.5 is interpreted as FAKE.
9. Probability < 0.5 is interpreted as REAL.


# ==================================================
# BASIC CNN
# ==================================================

The Basic CNN performs image-level deepfake detection.

It learns image-level visual patterns from the training
dataset and classifies an input face as REAL or FAKE.

Current verified results:

Accuracy: 67.34%
Precision: 69.44%
Recall: 62.50%
F1-score: 65.79%
ROC-AUC: 74.07%

The Basic CNN is currently the stronger baseline model
in the STREESHIELD experiment.


# ==================================================
# VIDEO DETECTION
# ==================================================

The video detection pipeline uses the 3D CNN.

The system extracts face frames from an uploaded video.

The final model input contains:

16 frames
128 x 128 pixels
3 RGB channels

Expected model input shape:

(16, 128, 128, 3)

If normal interval sampling does not produce enough
usable face frames, the preprocessing pipeline performs
denser fallback scanning to obtain 16 usable face frames
when possible.


# ==================================================
# 3D CNN
# ==================================================

The 3D CNN performs video-based deepfake detection.

Unlike a normal 2D CNN, the 3D CNN processes spatial and
temporal information.

Spatial information:
What the face looks like.

Temporal information:
How the face changes across frames.

Current verified results:

Accuracy: 50.00%
Precision: 50.00%
Recall: 100.00%
F1-score: 66.67%
ROC-AUC: 47.00%

Important limitation:

The current 3D CNN predicted every test sequence as FAKE.

Therefore, its 100% recall should not be interpreted as
evidence that it is better overall than the Basic CNN.


# ==================================================
# OPENCV
# ==================================================

OpenCV is used for:

- Image loading
- Video loading
- Face detection
- Frame extraction
- Face cropping
- Image resizing
- BGR to RGB conversion
- Video preprocessing


# ==================================================
# PREPROCESSING
# ==================================================

Image preprocessing:

- Detect face
- Select largest face
- Crop face
- Resize to 128 x 128
- Convert BGR to RGB
- Normalize pixels by dividing by 255

Video preprocessing:

- Read video frames
- Detect faces
- Extract usable face frames
- Build a 16-frame sequence
- Resize each face to 128 x 128
- Normalize pixel values


# ==================================================
# STREAMLIT APPLICATION
# ==================================================

Streamlit provides the STREESHIELD user interface.

The application supports:

- Image Detection
- Video Detection
- AI Analysis
- Risk Assessment
- AI Assistant
- AI Reports
- About / project information

The interface also supports theme-aware presentation
for light and dark modes.


# ==================================================
# AI ANALYSIS
# ==================================================

The AI Analysis layer receives an existing model result.

It uses:

- Model name
- Prediction
- Confidence
- Media type
- Raw model probability
- Media information

The analysis layer generates:

- Prediction explanation
- Confidence interpretation
- Possible manipulation indicators

Important:

Possible indicators are general model-oriented
interpretations. They do not prove that a specific
artifact exists in the uploaded media.


# ==================================================
# CONFIDENCE
# ==================================================

Confidence represents how strongly the model output
favors the predicted class.

High confidence indicates a stronger model output.

A confidence value near 50% indicates uncertainty.

Confidence is not a guarantee of correctness.


# ==================================================
# RISK ASSESSMENT
# ==================================================

STREESHIELD includes a risk-assessment layer that
interprets the model result and confidence.

Possible levels:

LOW
MEDIUM
HIGH

Risk level is an application-level interpretation and
should not be treated as absolute proof of authenticity
or manipulation.


# ==================================================
# AI ASSISTANT
# ==================================================

The STREESHIELD AI Assistant can answer questions about:

- Deepfake detection
- Basic CNN
- 3D CNN
- OpenCV
- Image preprocessing
- Video preprocessing
- Confidence scores
- Risk assessment
- AI analysis
- Streamlit
- STREESHIELD methodology
- Model limitations
- Current experimental results


# ==================================================
# DETECTION QUESTIONS
# ==================================================

Examples of questions the assistant can answer:

Why was this image classified as REAL?

Why was this image classified as FAKE?

What does confidence mean?

How does the Basic CNN work?

How does the 3D CNN work?

Why does the 3D CNN use 16 frames?

What does OpenCV do?

How does STREESHIELD detect deepfakes?

Which model is better?

What are the limitations of the current system?


# ==================================================
# PROJECT COMPARISON
# ==================================================

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

Overall baseline winner:

Basic CNN

The 3D CNN has higher recall and slightly higher F1-score,
but its accuracy and ROC-AUC are lower, and it predicts all
test sequences as FAKE.


# ==================================================
# ERROR ANALYSIS
# ==================================================

Basic CNN confusion matrix:

TN = 143
FP = 55
FN = 75
TP = 125

Basic CNN may produce false positives when REAL images
contain unusual lighting, blur, compression artifacts,
or patterns that differ from the training distribution.

False negatives may occur when manipulated images are
high quality or contain subtle artifacts.


3D CNN confusion matrix:

TN = 0
FP = 10
FN = 0
TP = 10

This reflects the model's tendency to classify every test
sequence as FAKE.


# ==================================================
# LIMITATIONS
# ==================================================

The current project has several limitations.

1. The video dataset is small compared with the scale
   needed for robust video deepfake detection.

2. The 3D CNN shows strong bias toward the FAKE class.

3. Model confidence is not perfectly calibrated.

4. A single model prediction cannot prove that a specific
   forensic artifact exists.

5. Real-world media can differ substantially from the
   training distribution.


# ==================================================
# FUTURE SCOPE
# ==================================================

Possible improvements include:

- Larger video datasets
- Greater dataset diversity
- Transfer learning
- Improved temporal architectures
- Transformer-based video models
- Explainable AI visualizations
- Better confidence calibration
- More extensive real-world testing
- Cloud deployment
- Advanced forensic report generation


# ==================================================
# PROJECT WORKFLOW
# ==================================================

Dataset
    ↓
Preprocessing
    ↓
Image CNN / Video 3D CNN
    ↓
REAL / FAKE + Confidence
    ↓
AI Analysis
    ↓
Risk Assessment
    ↓
AI Report
    ↓
AI Assistant
    ↓
Streamlit Application
"""


def get_knowledge():
    """
    Return the complete STREESHIELD knowledge base.
    """

    return STREESHIELD_KNOWLEDGE


if __name__ == "__main__":

    knowledge = get_knowledge()

    print("\n========================================")
    print("       STREESHIELD KNOWLEDGE BASE")
    print("========================================")

    print(
        "Knowledge characters:",
        len(knowledge)
    )

    print(
        "Knowledge loaded:",
        bool(knowledge.strip())
    )

    print("\n========================================")
    print("          12D TEST COMPLETE")
    print("========================================")