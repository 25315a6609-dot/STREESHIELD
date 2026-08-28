"""
STREESHIELD - Professional AI-Powered Deepfake Detection

Presentation layer around the existing trained models and AI modules.
Does not retrain or modify the existing CNN/3D CNN models.
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="STREESHIELD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
PREPROCESSING_DIR = os.path.join(APP_DIR, "preprocessing")
AI_ANALYSIS_DIR = os.path.join(PROJECT_DIR, "ai_analysis")
AI_ASSISTANT_DIR = os.path.join(PROJECT_DIR, "ai_assistant")
MODEL_DIR = os.path.join(APP_DIR, "models")

for path in [PREPROCESSING_DIR, AI_ANALYSIS_DIR, AI_ASSISTANT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

BASIC_CNN_PATH = os.path.join(MODEL_DIR, "basic_cnn.keras")
CNN3D_PATH = os.path.join(MODEL_DIR, "3d_cnn.keras")


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    from video_processor import load_face_detector, process_video
except ImportError as error:
    st.error("Could not load video preprocessing module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

try:
    from ai_analyzer import STREESHIELDAnalyzer
except ImportError as error:
    st.error("Could not load AI Analysis module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

try:
    from ai_report import STREESHIELDReportGenerator
except ImportError as error:
    st.error("Could not load AI Report module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

try:
    from combined_report import STREESHIELDCombinedReport
except ImportError as error:
    st.error("Could not load Combined Report module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

try:
    from risk_assesment import STREESHIELDRiskAssessment
except ImportError as error:
    st.error("Could not load Risk Assessment module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

try:
    from chat_interface import render_chat
except ImportError as error:
    st.error("Could not load AI Assistant module.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# ============================================================
# THEME-AWARE CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ss-text: var(--text-color);
        --ss-bg: var(--background-color);
        --ss-panel: var(--secondary-background-color);
        --ss-border: color-mix(in srgb, var(--text-color) 18%, transparent);
        --ss-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .ss-hero {
        padding: 2.4rem 1.5rem 2rem;
        border: 1px solid var(--ss-border);
        border-radius: 24px;
        text-align: center;
        background: linear-gradient(
            135deg,
            color-mix(in srgb, var(--ss-panel) 94%, #4f46e5 6%),
            var(--ss-panel)
        );
        margin-bottom: 1.4rem;
    }

    .ss-hero-icon {
        font-size: 3.2rem;
        line-height: 1;
        margin-bottom: .5rem;
    }

    .ss-hero-title {
        color: var(--ss-text);
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 850;
        letter-spacing: .12em;
        margin: 0;
    }

    .ss-hero-subtitle {
        color: var(--ss-text);
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: .45rem;
    }

    .ss-hero-tagline {
        color: var(--ss-muted);
        font-size: .92rem;
        margin-top: .55rem;
        letter-spacing: .08em;
    }

    .ss-card {
        border: 1px solid var(--ss-border);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        background: var(--ss-panel);
        min-height: 112px;
    }

    .ss-card-label {
        color: var(--ss-muted);
        font-size: .74rem;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-weight: 700;
    }

    .ss-card-value {
        color: var(--ss-text);
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: .35rem;
    }

    .ss-result {
        border: 1px solid var(--ss-border);
        border-radius: 22px;
        padding: 1.7rem;
        background: var(--ss-panel);
        text-align: center;
        margin: 1rem 0;
    }

    .ss-result-label {
        color: var(--ss-text);
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 900;
        margin: .4rem 0;
    }

    .ss-result-confidence {
        color: var(--ss-muted);
        font-size: 1.05rem;
        font-weight: 700;
    }

    .ss-section-note {
        color: var(--ss-muted);
        margin-bottom: .9rem;
    }

    .ss-about-box {
        border: 1px solid var(--ss-border);
        border-radius: 18px;
        padding: 1.25rem;
        background: var(--ss-panel);
    }

    .ss-footer {
        color: var(--ss-muted);
        text-align: center;
        padding-top: 1.5rem;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None

if "image_detection" not in st.session_state:
    st.session_state.image_detection = None

if "video_detection" not in st.session_state:
    st.session_state.video_detection = None


# ============================================================
# HELPERS
# ============================================================

def reset_application():
    st.session_state.reset_counter += 1
    st.session_state.last_detection = None
    st.session_state.image_detection = None
    st.session_state.video_detection = None


def info_card(label, value):
    st.markdown(
        f"""
        <div class="ss-card">
            <div class="ss-card-label">{label}</div>
            <div class="ss-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(prediction, confidence):
    icon = "🟢" if prediction == "REAL" else "🔴"
    st.markdown(
        f"""
        <div class="ss-result">
            <div class="ss-card-label">Detection Result</div>
            <div class="ss-result-label">{icon} {prediction}</div>
            <div class="ss-result-confidence">{confidence:.2f}% Confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_prediction(probability):
    probability = float(probability)
    if probability >= 0.5:
        return "FAKE", probability * 100
    return "REAL", (1.0 - probability) * 100


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    analyzer,
    model_name,
    prediction,
    confidence,
    media_type,
):
    """Convert the existing model result into the Phase 11 AI analysis."""

    result = analyzer.analyze_prediction(
        model_name=model_name,
        prediction=prediction,
        confidence=confidence,
        media_type=media_type,
    )

    explanation = (
        result.get("explanation", "")
        + " "
        + result.get("interpretation", "")
    ).strip()

    return explanation, result


def preprocess_image(image_bytes, detector):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded image could not be read.")

    height, width = image.shape[:2]

    # Preserve Phase 5 input distribution for already processed images.
    if height == 128 and width == 128:
        face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        return np.expand_dims(face, axis=0), face

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
    )

    if len(faces) == 0:
        return None, None

    largest_face = max(faces, key=lambda face: face[2] * face[3])
    x, y, w, h = largest_face
    face = image[y:y + h, x:x + w]
    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0

    return np.expand_dims(face, axis=0), face


def build_detection_result(
    media_type,
    model,
    prediction,
    confidence,
    probability,
    risk,
    risk_description,
    ai_explanation,
    analysis_result,
    report,
    **extra,
):
    result = {
        "media_type": media_type,
        "model": model,
        "prediction": prediction,
        "confidence": confidence,
        "probability": probability,
        "risk": risk,
        "risk_description": risk_description,
        "ai_explanation": ai_explanation,
        "analysis_result": analysis_result,
        "report": report,
    }
    result.update(extra)
    return result


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_basic_cnn():
    if not os.path.exists(BASIC_CNN_PATH):
        raise FileNotFoundError(BASIC_CNN_PATH)
    return tf.keras.models.load_model(BASIC_CNN_PATH)


@st.cache_resource
def load_cnn3d():
    if not os.path.exists(CNN3D_PATH):
        raise FileNotFoundError(CNN3D_PATH)
    return tf.keras.models.load_model(CNN3D_PATH)


@st.cache_resource
def get_face_detector():
    return load_face_detector()


@st.cache_resource
def load_ai_modules():
    return (
        STREESHIELDAnalyzer(),
        STREESHIELDReportGenerator(),
        STREESHIELDCombinedReport(),
        STREESHIELDRiskAssessment(),
    )


try:
    with st.spinner("Loading STREESHIELD system..."):
        basic_cnn = load_basic_cnn()
        cnn3d = load_cnn3d()
        face_detector = get_face_detector()
        analyzer, report_generator, combined_report_generator, risk_assessment = load_ai_modules()
except Exception as error:
    st.error("STREESHIELD could not initialize.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="ss-hero">
        <div class="ss-hero-icon">🛡️</div>
        <div class="ss-hero-title">STREESHIELD</div>
        <div class="ss-hero-subtitle">AI-Powered Deepfake Detection</div>
        <div class="ss-hero-tagline">Detect • Analyze • Explain • Understand</div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary1, summary2, summary3 = st.columns(3)
with summary1:
    info_card("Detection Engine", "CNN + 3D CNN")
with summary2:
    info_card("AI Layer", "Analysis + Assistant")
with summary3:
    if st.button("🔄 Reset", width="stretch"):
        reset_application()
        st.rerun()


# ============================================================
# NAVIGATION
# ============================================================

detection_tab, analysis_tab, assistant_tab, report_tab, about_tab = st.tabs(
    [
        "🎯 Detection",
        "🤖 AI Analysis",
        "🧠 AI Assistant",
        "📄 Report",
        "ℹ️ About",
    ]
)


# ============================================================
# DETECTION TAB
# ============================================================

with detection_tab:

    st.subheader("🎯 Deepfake Detection")
    st.markdown(
        '<div class="ss-section-note">Choose a media type and upload content for analysis.</div>',
        unsafe_allow_html=True,
    )

    image_tab, video_tab = st.tabs(["🖼️ Image", "🎥 Video"])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_tab:
        st.write("Upload a clear image containing a visible face.")

        image_key = f"image_upload_{st.session_state.reset_counter}"
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key=image_key,
        )

        if uploaded_image is not None:
            preview, details = st.columns([1.35, 0.65])

            with preview:
                st.image(
                    uploaded_image,
                    caption="Uploaded Image",
                    width="stretch",
                )

            with details:
                info_card("File", uploaded_image.name)
                info_card("Type", uploaded_image.type or "Image")
                info_card("Size", f"{uploaded_image.size / 1024:.1f} KB")

            try:
                progress = st.progress(0, text="Starting image analysis...")

                image_input, detected_face = preprocess_image(
                    uploaded_image.getvalue(),
                    face_detector,
                )

                if image_input is None:
                    progress.progress(100, text="No face detected")
                    st.warning(
                        "No face was detected. Upload an image with a clear, visible face."
                    )
                else:
                    progress.progress(30, text="Face prepared...")

                    face_col, input_col = st.columns(2)
                    with face_col:
                        st.image(
                            detected_face,
                            caption="Processed Face",
                            width=300,
                        )
                    with input_col:
                        info_card("Model", "Basic CNN")
                        info_card("Input", str(tuple(image_input.shape)))
                        info_card(
                            "Normalization",
                            f"{image_input.min():.3f} – {image_input.max():.3f}",
                        )

                    progress.progress(55, text="Running Basic CNN...")
                    probability = float(
                        basic_cnn.predict(image_input, verbose=0)[0][0]
                    )
                    prediction, confidence = format_prediction(probability)

                    progress.progress(70, text="Generating AI analysis...")
                    ai_explanation, analysis_result = generate_ai_explanation(
                        analyzer,
                        "Basic CNN",
                        prediction,
                        confidence,
                        "Image",
                    )

                    progress.progress(82, text="Calculating risk...")
                    risk = risk_assessment.calculate_risk(prediction, confidence)
                    risk_description = risk_assessment.get_description(risk)

                    report = report_generator.generate_report(
                        detection_type="Image",
                        model_name="Basic CNN",
                        prediction=prediction,
                        confidence=confidence,
                        ai_explanation=ai_explanation,
                    )

                    result_data = build_detection_result(
                        media_type="Image",
                        model="Basic CNN",
                        prediction=prediction,
                        confidence=confidence,
                        probability=probability,
                        risk=risk,
                        risk_description=risk_description,
                        ai_explanation=ai_explanation,
                        analysis_result=analysis_result,
                        report=report,
                        image_name=uploaded_image.name,
                        input_shape=tuple(image_input.shape),
                    )

                    st.session_state.image_detection = result_data
                    st.session_state.last_detection = result_data

                    progress.progress(100, text="Image analysis complete ✓")
                    result_card(prediction, confidence)

            except Exception as error:
                st.error("Image detection failed.")
                with st.expander("Technical details"):
                    st.code(str(error))

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    with video_tab:
        st.write(
            "Upload a video. STREESHIELD extracts a 16-frame face sequence and uses the 3D CNN."
        )

        video_key = f"video_upload_{st.session_state.reset_counter}"
        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            key=video_key,
        )

        if uploaded_video is not None:
            st.video(uploaded_video)

            temp_video_path = os.path.join(
                APP_DIR,
                "temp_uploaded_video_professional.mp4",
            )

            try:
                progress = st.progress(0, text="Starting video analysis...")

                progress.progress(15, text="Preparing video...")
                with open(temp_video_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                progress.progress(35, text="Extracting frames and detecting faces...")
                sequence, metadata = process_video(
                    temp_video_path,
                    face_detector,
                )

                progress.progress(52, text="Building 16-frame sequence...")
                video_input = np.expand_dims(sequence, axis=0)

                expected_shape = (1, 16, 128, 128, 3)
                if video_input.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected video input shape: {video_input.shape}; "
                        f"expected {expected_shape}"
                    )

                with st.expander("🎞️ Video Processing Details"):
                    v1, v2, v3, v4 = st.columns(4)
                    with v1:
                        st.metric("Frames", metadata["total_video_frames"])
                    with v2:
                        st.metric("Sampled", metadata["sampled_frames"])
                    with v3:
                        st.metric("Faces", metadata["faces_detected"])
                    with v4:
                        st.metric("Sequence", "16")

                progress.progress(65, text="Running 3D CNN...")
                probability = float(
                    cnn3d.predict(video_input, verbose=0)[0][0]
                )
                prediction, confidence = format_prediction(probability)

                progress.progress(76, text="Generating AI analysis...")
                ai_explanation, analysis_result = generate_ai_explanation(
                    analyzer,
                    "3D CNN",
                    prediction,
                    confidence,
                    "Video",
                )

                progress.progress(86, text="Calculating risk...")
                risk = risk_assessment.calculate_risk(prediction, confidence)
                risk_description = risk_assessment.get_description(risk)

                report = report_generator.generate_report(
                    detection_type="Video",
                    model_name="3D CNN",
                    prediction=prediction,
                    confidence=confidence,
                    ai_explanation=ai_explanation,
                )

                result_data = build_detection_result(
                    media_type="Video",
                    model="3D CNN",
                    prediction=prediction,
                    confidence=confidence,
                    probability=probability,
                    risk=risk,
                    risk_description=risk_description,
                    ai_explanation=ai_explanation,
                    analysis_result=analysis_result,
                    report=report,
                    video_name=uploaded_video.name,
                    metadata=metadata,
                    input_shape=tuple(video_input.shape),
                )

                st.session_state.video_detection = result_data
                st.session_state.last_detection = result_data

                progress.progress(100, text="Video analysis complete ✓")
                result_card(prediction, confidence)

            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error("Video detection failed.")
                with st.expander("Technical details"):
                    st.code(str(error))
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
        st.info("Run an image or video detection first to view its AI analysis.")
    else:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            info_card("Prediction", current["prediction"])
        with a2:
            info_card("Confidence", f"{current['confidence']:.2f}%")
        with a3:
            info_card("Risk", current["risk"])
        with a4:
            info_card("Model", current["model"])

        st.progress(
            min(max(current["confidence"] / 100, 0.0), 1.0),
            text=f"Model confidence: {current['confidence']:.2f}%",
        )

        st.markdown("### AI Interpretation")
        st.info(current["ai_explanation"])

        analysis_result = current["analysis_result"]

        st.markdown("### Confidence Interpretation")
        st.write(analysis_result.get("confidence_level", "Not available"))

        st.markdown("### Possible Indicators")
        for indicator in analysis_result.get("possible_indicators", []):
            st.write(f"✓ {indicator}")

        st.markdown("### Risk Assessment")
        if current["risk"] == "HIGH":
            st.error(current["risk_description"])
        elif current["risk"] == "MEDIUM":
            st.warning(current["risk_description"])
        else:
            st.success(current["risk_description"])

        st.caption(
            "These indicators are general model-oriented interpretations. "
            "They do not prove that a specific manipulation artifact exists."
        )


# ============================================================
# AI ASSISTANT TAB
# ============================================================

with assistant_tab:
    st.subheader("🧠 STREESHIELD AI Assistant")
    st.caption(
        "Ask about your detection result, deepfakes, CNN, 3D CNN, OpenCV, confidence, or STREESHIELD."
    )

    current = st.session_state.last_detection
    if current is not None:
        st.info(
            f"Current detection: {current['prediction']} • "
            f"{current['confidence']:.2f}% confidence • "
            f"{current['model']} • {current['media_type']}"
        )

    st.markdown("### Suggested questions")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.button("What is CNN?", width="stretch", disabled=True)
    with q2:
        st.button("Why was it classified this way?", width="stretch", disabled=True)
    with q3:
        st.button("What does confidence mean?", width="stretch", disabled=True)
    with q4:
        st.button("How does 3D CNN work?", width="stretch", disabled=True)

    render_chat()


# ============================================================
# REPORT TAB
# ============================================================

with report_tab:
    st.subheader("📄 AI Detection Report")

    current = st.session_state.last_detection

    if current is None:
        st.info("Run a detection first to generate a report.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            info_card("Media Type", current["media_type"])
        with r2:
            info_card("Model", current["model"])
        with r3:
            info_card("Prediction", current["prediction"])
        with r4:
            info_card("Risk", current["risk"])

        st.markdown("### Confidence")
        st.metric("Detection Confidence", f"{current['confidence']:.2f}%")

        st.markdown("### AI Interpretation")
        st.write(current["ai_explanation"])

        st.markdown("### Generated Report")
        st.text_area(
            "Generated Report",
            current["report"],
            height=300,
            label_visibility="collapsed",
        )

        st.download_button(
            "⬇️ Download AI Report",
            data=current["report"],
            file_name="streesheild_detection_report.txt",
            mime="text/plain",
            width="stretch",
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
                    ),
                )

                st.text_area(
                    "Combined Report",
                    combined,
                    height=320,
                )

                st.download_button(
                    "⬇️ Download Combined Report",
                    data=combined,
                    file_name="streesheild_combined_report.txt",
                    mime="text/plain",
                    width="stretch",
                    key="combined_report_download",
                )
            except Exception as error:
                st.warning(f"Combined report could not be generated: {error}")


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:
    st.subheader("ℹ️ About STREESHIELD")

    st.markdown(
        """
        <div class="ss-about-box">
            <strong>STREESHIELD</strong> combines image-based and
            video-based deepfake detection with AI-assisted
            interpretation, risk assessment, reporting, and a
            conversational assistant.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System Workflow")
    workflow_cols = st.columns(5)
    workflow = [
        ("🖼️", "Image", "Basic CNN"),
        ("🎥", "Video", "3D CNN"),
        ("🤖", "Analysis", "Explanation"),
        ("⚠️", "Risk", "Assessment"),
        ("📄", "Report", "AI Report"),
    ]

    for col, (icon, name, detail) in zip(workflow_cols, workflow):
        with col:
            st.markdown(f"### {icon}")
            st.write(f"**{name}**")
            st.caption(detail)

    st.markdown("### Current Baseline Results")
    st.dataframe(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "Basic CNN": ["67.34%", "69.44%", "62.50%", "65.79%", "74.07%"],
            "3D CNN": ["50.00%", "50.00%", "100.00%", "66.67%", "47.00%"],
        },
        hide_index=True,
        width="stretch",
    )

    st.warning(
        "The current 3D CNN is a weak baseline and predicted all test video sequences as FAKE. "
        "Its 100% recall should therefore not be interpreted as superior overall performance."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🛡️ STREESHIELD • AI-Powered Deepfake Detection • Detect • Analyze • Explain • Understand"
)
