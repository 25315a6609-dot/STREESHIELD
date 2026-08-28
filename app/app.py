"""
STREESHIELD PHASE 13D
AI Layer + Streamlit Integration

Clean Streamlit-native UI.

This version preserves:
- Basic CNN
- 3D CNN
- Face detection
- Image preprocessing
- Video preprocessing
- AI Analysis
- Risk Assessment
- AI Reports
- AI Assistant

The UI does NOT use visible HTML <div> elements.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="STREESHIELD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    APP_DIR
)

PREPROCESSING_DIR = os.path.join(
    APP_DIR,
    "preprocessing"
)

AI_ANALYSIS_DIR = os.path.join(
    PROJECT_DIR,
    "ai_analysis"
)

AI_ASSISTANT_DIR = os.path.join(
    PROJECT_DIR,
    "ai_assistant"
)


# ============================================================
# PYTHON PATH
# ============================================================

for path in [

    PREPROCESSING_DIR,
    AI_ANALYSIS_DIR,
    AI_ASSISTANT_DIR

]:

    if path not in sys.path:

        sys.path.insert(
            0,
            path
        )


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

try:

    from video_processor import (
        load_face_detector,
        process_video
    )

    from prediction_display import (
        display_prediction,
        display_processing_info
    )

except ImportError as error:

    st.error(
        "Could not load preprocessing modules."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMPORT AI ANALYSIS
# ============================================================

try:

    from ai_analyzer import (
        STREESHIELDAnalyzer
    )

except ImportError as error:

    st.error(
        "Could not load AI Analysis module."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMPORT AI REPORT
# ============================================================

try:

    from ai_report import (
        STREESHIELDReportGenerator
    )

except ImportError as error:

    st.error(
        "Could not load AI Report module."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMPORT COMBINED REPORT
# ============================================================

try:

    from combined_report import (
        STREESHIELDCombinedReport
    )

except ImportError as error:

    st.error(
        "Could not load Combined Report module."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMPORT RISK ASSESSMENT
# ============================================================

try:

    from risk_assesment import (
        STREESHIELDRiskAssessment
    )

except ImportError as error:

    st.error(
        "Could not load Risk Assessment module."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMPORT AI ASSISTANT
# ============================================================

try:

    from chat_interface import (
        render_chat
    )

except ImportError as error:

    st.error(
        "Could not load AI Assistant."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = os.path.join(
    APP_DIR,
    "models"
)

BASIC_CNN_PATH = os.path.join(
    MODEL_DIR,
    "basic_cnn.keras"
)

CNN3D_PATH = os.path.join(
    MODEL_DIR,
    "3d_cnn.keras"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "reset_counter_13d" not in st.session_state:

    st.session_state.reset_counter_13d = 0


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_application():

    st.session_state.reset_counter_13d += 1

    keys_to_remove = [

        "detection_mode_13d"

    ]

    for key in keys_to_remove:

        if key in st.session_state:

            del st.session_state[key]

    st.rerun()


# ============================================================
# LOAD BASIC CNN
# ============================================================

@st.cache_resource
def load_basic_cnn():

    if not os.path.exists(
        BASIC_CNN_PATH
    ):

        raise FileNotFoundError(
            "Basic CNN model not found:\n"
            f"{BASIC_CNN_PATH}"
        )

    return tf.keras.models.load_model(
        BASIC_CNN_PATH
    )


# ============================================================
# LOAD 3D CNN
# ============================================================

@st.cache_resource
def load_cnn3d():

    if not os.path.exists(
        CNN3D_PATH
    ):

        raise FileNotFoundError(
            "3D CNN model not found:\n"
            f"{CNN3D_PATH}"
        )

    return tf.keras.models.load_model(
        CNN3D_PATH
    )


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def get_face_detector():

    return load_face_detector()


# ============================================================
# LOAD AI MODULES
# ============================================================

@st.cache_resource
def load_ai_modules():

    analyzer = STREESHIELDAnalyzer()

    report_generator = (
        STREESHIELDReportGenerator()
    )

    combined_report_generator = (
        STREESHIELDCombinedReport()
    )

    risk_assessment = (
        STREESHIELDRiskAssessment()
    )

    return (

        analyzer,
        report_generator,
        combined_report_generator,
        risk_assessment

    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_bytes,
    detector
):

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "The uploaded image could not be read."
        )

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # ALREADY PROCESSED IMAGE
    # --------------------------------------------------------

    if height == 128 and width == 128:

        face = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        face = (
            face.astype(
                np.float32
            ) / 255.0
        )

        model_input = np.expand_dims(
            face,
            axis=0
        )

        return model_input, face

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector.detectMultiScale(

        gray,

        scaleFactor=1.05,

        minNeighbors=3,

        minSize=(20, 20)

    )

    if len(faces) == 0:

        return None, None

    # --------------------------------------------------------
    # LARGEST FACE
    # --------------------------------------------------------

    largest_face = max(

        faces,

        key=lambda face:
        face[2] * face[3]

    )

    x, y, w, h = largest_face

    # --------------------------------------------------------
    # CROP FACE
    # --------------------------------------------------------

    face = image[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    face = cv2.resize(
        face,
        (128, 128)
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    face = (
        face.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # BATCH DIMENSION
    # --------------------------------------------------------

    model_input = np.expand_dims(
        face,
        axis=0
    )

    return model_input, face


# ============================================================
# FORMAT PREDICTION
# ============================================================

def format_prediction(
    probability
):

    probability = float(
        probability
    )

    if probability >= 0.5:

        prediction = "FAKE"

        confidence = (
            probability * 100
        )

    else:

        prediction = "REAL"

        confidence = (
            (1.0 - probability) * 100
        )

    return (
        prediction,
        confidence
    )


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(

    analyzer,
    model_name,
    prediction,
    confidence,
    media_type

):

    result = analyzer.analyze_prediction(

        model_name=model_name,

        prediction=prediction,

        confidence=confidence,

        media_type=media_type

    )

    explanation = (

        result["explanation"]
        + " "
        + result["interpretation"]

    )

    return (
        explanation,
        result
    )


# ============================================================
# DISPLAY RISK
# ============================================================

def display_risk(
    risk,
    description
):

    st.subheader(
        "⚠️ Risk Assessment"
    )

    if risk == "HIGH":

        st.error(
            f"Risk Level: {risk}"
        )

    elif risk == "MEDIUM":

        st.warning(
            f"Risk Level: {risk}"
        )

    else:

        st.success(
            f"Risk Level: {risk}"
        )

    st.write(
        description
    )


# ============================================================
# DISPLAY AI EXPLANATION
# ============================================================

def display_ai_explanation(

    explanation,
    analysis_result

):

    st.subheader(
        "🤖 AI Analysis"
    )

    st.info(
        explanation
    )

    with st.expander(
        "View AI Analysis Details"
    ):

        st.write(
            "Model:",
            analysis_result["model"]
        )

        st.write(
            "Media Type:",
            analysis_result["media_type"]
        )

        st.write(
            "Prediction:",
            analysis_result["prediction"]
        )

        st.write(
            "Confidence:",
            f"{analysis_result['confidence']:.2f}%"
        )

        st.write(
            "Confidence Level:",
            analysis_result["confidence_level"]
        )

        st.write(
            "Possible Indicators:"
        )

        for indicator in analysis_result[
            "possible_indicators"
        ]:

            st.write(
                f"✓ {indicator}"
            )


# ============================================================
# DISPLAY INDIVIDUAL REPORT
# ============================================================

def display_individual_report(

    report_generator,
    media_type,
    model_name,
    prediction,
    confidence,
    explanation

):

    report = report_generator.generate_report(

        detection_type=media_type,

        model_name=model_name,

        prediction=prediction,

        confidence=confidence,

        ai_explanation=explanation

    )

    st.subheader(
        "📄 AI Detection Report"
    )

    st.text_area(
        "Generated Report",
        report,
        height=250
    )


# ============================================================
# DISPLAY COMBINED REPORT
# ============================================================

def display_combined_report(

    combined_generator,

    cnn_prediction,
    cnn_confidence,

    cnn3d_prediction,
    cnn3d_confidence,

    explanation

):

    report = combined_generator.generate_report(

        cnn_prediction=cnn_prediction,

        cnn_confidence=cnn_confidence,

        cnn3d_prediction=cnn3d_prediction,

        cnn3d_confidence=cnn3d_confidence,

        ai_explanation=explanation

    )

    st.subheader(
        "📑 Combined AI Report"
    )

    st.text_area(
        "Combined Report",
        report,
        height=300
    )




# ============================================================
# MODERN STREESHIELD UI
# Presentation layer only. Existing detection functions above
# remain unchanged.
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero-box {
        padding: 2.3rem 2rem;
        border-radius: 24px;
        text-align: center;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        margin-bottom: 1.25rem;
    }

    .hero-icon {
        font-size: 3.4rem;
        line-height: 1;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 850;
        letter-spacing: 0.16rem;
        color: #ffffff;
        margin-top: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #cbd5e1;
        margin-top: 0.4rem;
    }

    .hero-tagline {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 0.75rem;
        letter-spacing: 0.05rem;
    }

    .section-card {
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        margin-bottom: 1rem;
    }

    .mini-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        opacity: 0.65;
    }

    .result-banner {
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin: 0.8rem 0 1.1rem 0;
    }

    .result-label {
        font-size: 2.5rem;
        font-weight: 850;
        margin: 0.35rem 0;
    }

    .result-confidence {
        font-size: 1.05rem;
        font-weight: 650;
    }

    .ai-panel {
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "reset_counter_13d" not in st.session_state:
    st.session_state.reset_counter_13d = 0

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None

if "image_detection" not in st.session_state:
    st.session_state.image_detection = None

if "video_detection" not in st.session_state:
    st.session_state.video_detection = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-icon">🛡️</div>
        <div class="hero-title">STREESHIELD</div>
        <div class="hero-subtitle">AI-Powered Deepfake Detection</div>
        <div class="hero-tagline">Detect • Analyze • Explain • Understand</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP SUMMARY + RESET
# ============================================================

summary_left, summary_mid, summary_right = st.columns([1, 1, 1])

with summary_left:
    st.metric("Detection Engine", "CNN + 3D CNN")

with summary_mid:
    st.metric("AI Layer", "Analysis + Assistant")

with summary_right:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.reset_counter_13d += 1
        st.session_state.last_detection = None
        st.session_state.image_detection = None
        st.session_state.video_detection = None
        st.rerun()


# ============================================================
# MAIN NAVIGATION
# ============================================================

detection_tab, analysis_tab, assistant_tab, report_tab, about_tab = st.tabs(
    [
        "🎯 Detection",
        "🤖 AI Analysis",
        "🧠 AI Assistant",
        "📄 Report",
        "ℹ️ About"
    ]
)


# ============================================================
# LOAD MODELS + AI MODULES
# ============================================================

try:
    with st.spinner("Loading STREESHIELD system..."):
        basic_cnn = load_basic_cnn()
        cnn3d = load_cnn3d()
        face_detector = get_face_detector()
        (
            analyzer,
            report_generator,
            combined_report_generator,
            risk_assessment
        ) = load_ai_modules()
except Exception as error:
    st.error("STREESHIELD initialization failed.")
    st.exception(error)
    st.stop()


# ============================================================
# DETECTION TAB
# ============================================================

with detection_tab:

    st.subheader("🎯 Deepfake Detection")
    st.caption(
        "Choose a media type, upload your file, and STREESHIELD will run the existing trained model."
    )

    image_tab, video_tab = st.tabs(["🖼️ Image", "🎥 Video"])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_tab:

        st.write("Upload a clear image containing a visible face.")

        image_key = (
            "image_upload_modern_"
            f"{st.session_state.reset_counter_13d}"
        )

        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key=image_key
        )

        if uploaded_image is not None:

            preview_col, info_col = st.columns([1.25, 0.75])

            with preview_col:
                st.image(
                    uploaded_image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            with info_col:
                st.markdown("### File Information")
                st.write("Name:", uploaded_image.name)
                st.write("Type:", uploaded_image.type)
                st.write(
                    "Size:",
                    f"{uploaded_image.size / 1024:.1f} KB"
                )

            try:
                progress = st.progress(0, text="Preparing image...")

                image_input, detected_face = preprocess_image(
                    uploaded_image.getvalue(),
                    face_detector
                )

                if image_input is None:
                    progress.progress(100, text="No face detected")
                    st.warning(
                        "No face was detected. Upload an image with a clear, visible face."
                    )
                else:
                    progress.progress(35, text="Face prepared...")

                    face_col, meta_col = st.columns([1, 1])

                    with face_col:
                        st.image(
                            detected_face,
                            caption="Processed Face",
                            width=300
                        )

                    with meta_col:
                        st.markdown("### Model Input")
                        st.write("Shape:", image_input.shape)
                        st.write(
                            "Pixel range:",
                            f"{image_input.min():.4f} – {image_input.max():.4f}"
                        )
                        st.write("Model:", "Basic CNN")

                    progress.progress(60, text="Running Basic CNN...")

                    probability = float(
                        basic_cnn.predict(
                            image_input,
                            verbose=0
                        )[0][0]
                    )

                    prediction, confidence = format_prediction(
                        probability
                    )

                    progress.progress(75, text="Generating AI analysis...")

                    ai_explanation, analysis_result = generate_ai_explanation(
                        analyzer,
                        "Basic CNN",
                        prediction,
                        confidence,
                        "Image"
                    )

                    progress.progress(85, text="Calculating risk...")

                    risk = risk_assessment.calculate_risk(
                        prediction,
                        confidence
                    )

                    risk_description = risk_assessment.get_description(
                        risk
                    )

                    report = report_generator.generate_report(
                        detection_type="Image",
                        model_name="Basic CNN",
                        prediction=prediction,
                        confidence=confidence,
                        ai_explanation=ai_explanation
                    )

                    result_data = {
                        "media_type": "Image",
                        "model": "Basic CNN",
                        "prediction": prediction,
                        "confidence": confidence,
                        "probability": probability,
                        "risk": risk,
                        "risk_description": risk_description,
                        "ai_explanation": ai_explanation,
                        "analysis_result": analysis_result,
                        "report": report,
                        "image_name": uploaded_image.name,
                        "input_shape": tuple(image_input.shape),
                    }

                    st.session_state.image_detection = result_data
                    st.session_state.last_detection = result_data

                    progress.progress(100, text="Image analysis complete ✓")

                    if prediction == "REAL":
                        st.success(
                            f"🟢 REAL • {confidence:.2f}% confidence"
                        )
                    else:
                        st.error(
                            f"🔴 FAKE • {confidence:.2f}% confidence"
                        )

                    st.info(
                        "Detailed AI Analysis, risk assessment, and the report are available in the tabs above."
                    )

            except Exception as error:
                st.error("Image detection failed.")
                st.exception(error)


    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    with video_tab:

        st.write(
            "Upload a video. STREESHIELD extracts a 16-frame face sequence and runs the existing 3D CNN."
        )

        video_key = (
            "video_upload_modern_"
            f"{st.session_state.reset_counter_13d}"
        )

        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            key=video_key
        )

        if uploaded_video is not None:

            st.video(uploaded_video)

            temp_video_path = os.path.join(
                APP_DIR,
                "temp_uploaded_video_modern.mp4"
            )

            try:
                progress = st.progress(0, text="Preparing video...")

                with open(temp_video_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                progress.progress(20, text="Extracting frames and detecting faces...")

                sequence, metadata = process_video(
                    temp_video_path,
                    face_detector
                )

                progress.progress(45, text="Building 16-frame sequence...")

                video_input = np.expand_dims(
                    sequence,
                    axis=0
                )

                expected_shape = (
                    1,
                    16,
                    128,
                    128,
                    3
                )

                if video_input.shape != expected_shape:
                    raise ValueError(
                        "Unexpected video input shape: "
                        f"{video_input.shape}; expected {expected_shape}"
                    )

                with st.expander("Video Processing Details", expanded=False):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Frames Sampled",
                            metadata["sampled_frames"]
                        )

                    with c2:
                        st.metric(
                            "Faces Detected",
                            metadata["faces_detected"]
                        )

                    with c3:
                        st.metric(
                            "Sequence",
                            "16 Frames"
                        )

                    st.write(
                        "Model input shape:",
                        video_input.shape
                    )

                progress.progress(65, text="Running 3D CNN...")

                probability = float(
                    cnn3d.predict(
                        video_input,
                        verbose=0
                    )[0][0]
                )

                prediction, confidence = format_prediction(
                    probability
                )

                progress.progress(78, text="Generating AI analysis...")

                ai_explanation, analysis_result = generate_ai_explanation(
                    analyzer,
                    "3D CNN",
                    prediction,
                    confidence,
                    "Video"
                )

                progress.progress(86, text="Calculating risk...")

                risk = risk_assessment.calculate_risk(
                    prediction,
                    confidence
                )

                risk_description = risk_assessment.get_description(
                    risk
                )

                report = report_generator.generate_report(
                    detection_type="Video",
                    model_name="3D CNN",
                    prediction=prediction,
                    confidence=confidence,
                    ai_explanation=ai_explanation
                )

                result_data = {
                    "media_type": "Video",
                    "model": "3D CNN",
                    "prediction": prediction,
                    "confidence": confidence,
                    "probability": probability,
                    "risk": risk,
                    "risk_description": risk_description,
                    "ai_explanation": ai_explanation,
                    "analysis_result": analysis_result,
                    "report": report,
                    "video_name": uploaded_video.name,
                    "metadata": metadata,
                    "input_shape": tuple(video_input.shape),
                }

                st.session_state.video_detection = result_data
                st.session_state.last_detection = result_data

                progress.progress(100, text="Video analysis complete ✓")

                if prediction == "REAL":
                    st.success(
                        f"🟢 REAL • {confidence:.2f}% confidence"
                    )
                else:
                    st.error(
                        f"🔴 FAKE • {confidence:.2f}% confidence"
                    )

                st.info(
                    "Detailed AI Analysis, risk assessment, and the report are available in the tabs above."
                )

            except ValueError as error:
                st.warning(str(error))

            except Exception as error:
                st.error("Video detection failed.")
                st.exception(error)

            finally:
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except OSError:
                        pass


# ============================================================
# AI ANALYSIS TAB
# ============================================================

with analysis_tab:

    st.subheader("🤖 AI Analysis")

    current = st.session_state.last_detection

    if current is None:
        st.info(
            "Run an image or video detection first. Your AI interpretation will appear here."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Prediction", current["prediction"])

        with c2:
            st.metric(
                "Confidence",
                f"{current['confidence']:.2f}%"
            )

        with c3:
            st.metric("Risk", current["risk"])

        with c4:
            st.metric("Model", current["model"])

        st.progress(
            min(max(current["confidence"] / 100, 0.0), 1.0),
            text=f"Model confidence: {current['confidence']:.2f}%"
        )

        st.markdown("### AI Interpretation")
        st.info(current["ai_explanation"])

        analysis_result = current["analysis_result"]

        st.markdown("### Confidence Interpretation")
        st.write(
            analysis_result["confidence_interpretation"]
        )

        st.markdown("### Possible Indicators")

        for indicator in analysis_result["possible_indicators"]:
            st.write(f"✓ {indicator}")

        st.markdown("### Risk Assessment")

        if current["risk"] == "HIGH":
            st.error(current["risk_description"])
        elif current["risk"] == "MEDIUM":
            st.warning(current["risk_description"])
        else:
            st.success(current["risk_description"])

        st.caption(
            "AI indicators are general model-oriented interpretations. "
            "They do not prove that a specific manipulation artifact exists."
        )


# ============================================================
# AI ASSISTANT TAB
# ============================================================

with assistant_tab:

    st.subheader("🧠 STREESHIELD AI Assistant")

    st.caption(
        "Ask about the current detection or about STREESHIELD, CNN, 3D CNN, OpenCV, confidence, and deepfakes."
    )

    current = st.session_state.last_detection

    if current is not None:
        detection_context = (
            f"Current detection: {current['prediction']} "
            f"with {current['confidence']:.2f}% confidence using "
            f"{current['model']} on {current['media_type']}."
        )
        st.info(detection_context)

    # Existing chat component remains unchanged.
    render_chat()


# ============================================================
# REPORT TAB
# ============================================================

with report_tab:

    st.subheader("📄 AI Detection Report")

    current = st.session_state.last_detection

    if current is None:
        st.info(
            "Run a detection first to generate a report."
        )
    else:
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.write("**Media Type**")
            st.write(current["media_type"])

        with r2:
            st.write("**Model**")
            st.write(current["model"])

        with r3:
            st.write("**Prediction**")
            st.write(current["prediction"])

        with r4:
            st.write("**Risk**")
            st.write(current["risk"])

        st.divider()

        st.markdown("### Confidence")
        st.metric(
            "Detection Confidence",
            f"{current['confidence']:.2f}%"
        )

        st.markdown("### AI Interpretation")
        st.write(current["ai_explanation"])

        st.markdown("### Generated Report")
        st.text_area(
            "Report",
            current["report"],
            height=320,
            label_visibility="collapsed"
        )

        st.download_button(
            "⬇️ Download AI Report",
            data=current["report"],
            file_name="streesheild_detection_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        if (
            st.session_state.image_detection is not None
            and st.session_state.video_detection is not None
        ):
            st.divider()
            st.markdown("### Combined CNN + 3D CNN Report")

            image_result = st.session_state.image_detection
            video_result = st.session_state.video_detection

            try:
                combined = combined_report_generator.generate_report(
                    cnn_prediction=image_result["prediction"],
                    cnn_confidence=image_result["confidence"],
                    cnn3d_prediction=video_result["prediction"],
                    cnn3d_confidence=video_result["confidence"],
                    ai_explanation=(
                        image_result["ai_explanation"]
                        + " | "
                        + video_result["ai_explanation"]
                    )
                )

                st.text_area(
                    "Combined Report",
                    combined,
                    height=320
                )

                st.download_button(
                    "⬇️ Download Combined Report",
                    data=combined,
                    file_name="streesheild_combined_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_combined_report"
                )

            except Exception as error:
                st.warning(
                    f"Combined report could not be generated: {error}"
                )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    st.subheader("ℹ️ About STREESHIELD")

    st.write(
        "STREESHIELD combines image-based and video-based "
        "deepfake detection with AI-assisted interpretation."
    )

    st.markdown("### Detection Workflow")

    workflow = [
        "🖼️ Image → Basic CNN → REAL / FAKE",
        "🎥 Video → 16-frame sequence → 3D CNN → REAL / FAKE",
        "🤖 Prediction → AI Analysis → Risk Assessment",
        "🧠 AI Assistant → User questions and explanations",
        "📄 Detection → AI Report"
    ]

    for item in workflow:
        st.write(item)

    st.markdown("### Current Baseline Results")

    result_table = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ],
        "Basic CNN": [
            "67.34%",
            "69.44%",
            "62.50%",
            "65.79%",
            "74.07%"
        ],
        "3D CNN": [
            "50.00%",
            "50.00%",
            "100.00%",
            "66.67%",
            "47.00%"
        ]
    }

    st.table(result_table)

    st.warning(
        "The current 3D CNN is a weak baseline and predicted all test video sequences as FAKE. "
        "Its 100% recall therefore should not be interpreted as superior overall performance."
    )

    st.markdown("### Project Modules")

    modules = st.columns(4)

    module_data = [
        ("🧠", "Basic CNN"),
        ("🎥", "3D CNN"),
        ("🤖", "AI Analysis"),
        ("🧠", "AI Assistant")
    ]

    for column, (icon, name) in zip(modules, module_data):
        with column:
            st.markdown(f"### {icon}")
            st.caption(name)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ STREESHIELD • AI-Powered Deepfake Detection • "
    "Detect • Analyze • Explain • Understand"
)
