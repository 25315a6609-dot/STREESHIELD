import streamlit as st

from openai_assistant import (
    OpenAISTREESHIELDAssistant
)


# ==================================================
# CHAT INTERFACE
# ==================================================

def render_chat(
    detection_context=None
):
    """
    Render the STREESHIELD AI Assistant.

    detection_context can contain the current
    image/video prediction, confidence, model,
    and risk information.
    """

    st.subheader(
        "🧠 STREESHIELD AI Assistant"
    )

    st.caption(
        "Ask about your detection result, deepfakes, "
        "CNN, 3D CNN, OpenCV, confidence, or "
        "STREESHIELD."
    )


    # ==================================================
    # SESSION STATE
    # ==================================================

    if "ai_chat_history" not in st.session_state:

        st.session_state.ai_chat_history = []


    # ==================================================
    # CURRENT DETECTION CONTEXT
    # ==================================================

    if detection_context:

        st.info(
            "Current detection: "
            f"{detection_context.get('prediction', 'N/A')} • "
            f"{detection_context.get('confidence', 0):.2f}% "
            "confidence • "
            f"{detection_context.get('model', 'N/A')} • "
            f"{detection_context.get('media_type', 'N/A')}"
        )


    # ==================================================
    # SUGGESTED QUESTIONS
    # ==================================================

    st.markdown(
        "#### Suggested Questions"
    )

    q1, q2, q3, q4 = st.columns(4)


    with q1:

        st.button(
            "What is CNN?",
            use_container_width=True,
            key="suggest_cnn",
            disabled=True
        )


    with q2:

        st.button(
            "What does confidence mean?",
            use_container_width=True,
            key="suggest_confidence",
            disabled=True
        )


    with q3:

        st.button(
            "How does 3D CNN work?",
            use_container_width=True,
            key="suggest_3dcnn",
            disabled=True
        )


    with q4:

        st.button(
            "Why was it classified this way?",
            use_container_width=True,
            key="suggest_prediction",
            disabled=True
        )


    st.divider()


    # ==================================================
    # DISPLAY CHAT HISTORY
    # ==================================================

    for message in st.session_state.ai_chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ==================================================
    # USER INPUT
    # ==================================================

    question = st.chat_input(
        "Ask STREESHIELD AI..."
    )


    if question:

        # ----------------------------------------------
        # SAVE USER MESSAGE
        # ----------------------------------------------

        st.session_state.ai_chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ----------------------------------------------
        # BUILD DETECTION CONTEXT
        # ----------------------------------------------

        context = None

        if detection_context:

            context = (
                f"Media type: "
                f"{detection_context.get('media_type', 'N/A')}\n"
                f"Model: "
                f"{detection_context.get('model', 'N/A')}\n"
                f"Prediction: "
                f"{detection_context.get('prediction', 'N/A')}\n"
                f"Confidence: "
                f"{detection_context.get('confidence', 0):.2f}%\n"
                f"Risk: "
                f"{detection_context.get('risk', 'N/A')}\n"
            )


        # ----------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------

        try:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "STREESHIELD AI is thinking..."
                ):

                    assistant = (
                        OpenAISTREESHIELDAssistant()
                    )

                    answer = assistant.ask(
                        question,
                        detection_context=context
                    )

                st.markdown(
                    answer
                )


            # ------------------------------------------
            # SAVE AI RESPONSE
            # ------------------------------------------

            st.session_state.ai_chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


        except Exception as error:

            with st.chat_message(
                "assistant"
            ):

                st.error(
                    "The AI Assistant could not respond."
                )

                st.caption(
                    str(error)
                )


    # ==================================================
    # CLEAR CONVERSATION
    # ==================================================

    if st.session_state.ai_chat_history:

        if st.button(
            "🗑️ Clear Conversation",
            key="clear_ai_chat"
        ):

            st.session_state.ai_chat_history = []

            st.rerun()