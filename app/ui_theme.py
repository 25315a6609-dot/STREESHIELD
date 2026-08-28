import streamlit as st


# ==================================================
# STREESHIELD UI THEME
# ==================================================

def apply_theme():
    """
    Apply presentation-only styling.
    No model or detection logic is included here.
    """

    st.markdown(
        """
        <style>
        
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .stButton > button {
            border-radius: 12px;
            min-height: 45px;
            font-weight: 600;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# HEADER
# ==================================================

def render_header():

    st.title("🛡️ STREESHIELD")

    st.subheader(
        "AI-Powered Deepfake Detection"
    )

    st.caption(
        "Detect • Analyze • Explain • Understand"
    )

    st.write(
        "An AI-assisted deepfake detection platform "
        "for images and videos."
    )

    st.divider()


# ==================================================
# SECTION TITLE
# ==================================================

def render_section_title(title):

    st.header(title)


# ==================================================
# INFO CARD
# ==================================================

def render_info_card(
    label,
    value
):

    st.metric(
        label=label,
        value=value
    )


# ==================================================
# RESULT CARD
# ==================================================

def render_result_card(
    prediction,
    confidence
):

    prediction = str(
        prediction
    ).upper()

    confidence = float(
        confidence
    )

    if prediction == "REAL":

        st.success(
            f"🟢 REAL\n\n"
            f"**{confidence:.2f}% Confidence**"
        )

    else:

        st.error(
            f"🔴 FAKE\n\n"
            f"**{confidence:.2f}% Confidence**"
        )


# ==================================================
# AI SECTION
# ==================================================

def render_ai_container(
    title="🤖 AI Analysis"
):

    st.subheader(title)


# ==================================================
# FOOTER
# ==================================================

def render_footer():

    st.divider()

    st.caption(
        "STREESHIELD • AI-Powered Deepfake Detection"
    )