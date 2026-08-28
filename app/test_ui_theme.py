import streamlit as st

from ui_theme import (
    apply_theme,
    render_header,
    render_section_title,
    render_info_card,
    render_result_card,
    render_ai_container,
    render_footer
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="STREESHIELD UI Preview",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# APPLY THEME
# ==================================================

apply_theme()


# ==================================================
# HEADER
# ==================================================

render_header()


# ==================================================
# DETECTION MODE
# ==================================================

render_section_title(
    "Detection Mode"
)

mode1, mode2 = st.columns(2)


with mode1:

    st.button(
        "🖼️ Image Detection",
        use_container_width=True
    )


with mode2:

    st.button(
        "🎥 Video Detection",
        use_container_width=True
    )


# ==================================================
# DETECTION OVERVIEW
# ==================================================

render_section_title(
    "Detection Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    render_info_card(
        "Model",
        "Basic CNN"
    )


with col2:

    render_info_card(
        "Media",
        "Image"
    )


with col3:

    render_info_card(
        "Risk",
        "Medium"
    )


with col4:

    render_info_card(
        "Status",
        "Analyzed"
    )


# ==================================================
# UPLOAD
# ==================================================

render_section_title(
    "Upload Media"
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )


# ==================================================
# RESULT
# ==================================================

render_section_title(
    "Detection Result"
)

render_result_card(
    "REAL",
    86.70
)


# ==================================================
# AI ANALYSIS
# ==================================================

render_ai_container(
    "🤖 AI Analysis"
)

ai1, ai2 = st.columns(2)


with ai1:

    render_info_card(
        "Prediction",
        "REAL"
    )


with ai2:

    render_info_card(
        "Confidence Level",
        "Moderate"
    )


st.write(
    "The model detected patterns that are more "
    "consistent with the REAL class."
)


st.markdown(
    "### Possible Indicators"
)

st.write(
    "✓ Facial texture consistency"
)

st.write(
    "✓ Natural image-level patterns"
)

st.write(
    "✓ No strong manipulation pattern detected"
)


# ==================================================
# AI ASSISTANT
# ==================================================

render_section_title(
    "🧠 STREESHIELD AI Assistant"
)

st.write(
    "Ask questions about this detection, deepfakes, "
    "CNN, 3D CNN, or the STREESHIELD methodology."
)


question = st.chat_input(
    "Ask STREESHIELD AI..."
)


if question:

    with st.chat_message("user"):

        st.write(question)


    with st.chat_message("assistant"):

        st.write(
            "This is the AI Assistant interface preview."
        )


# ==================================================
# REPORT
# ==================================================

render_section_title(
    "📄 Detection Report"
)

report_col1, report_col2 = st.columns(2)


with report_col1:

    st.write("Media Type")
    st.write("Image")

    st.write("Model")
    st.write("Basic CNN")

    st.write("Prediction")
    st.write("REAL")


with report_col2:

    st.write("Confidence")
    st.write("86.70%")

    st.write("Risk")
    st.write("Medium")


st.button(
    "📄 Generate Report"
)


# ==================================================
# FOOTER
# ==================================================

render_footer()