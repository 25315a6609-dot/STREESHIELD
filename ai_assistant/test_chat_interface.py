import streamlit as st

from chat_interface import render_chat


# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="STREESHIELD AI Assistant",
    page_icon="🤖"
)


st.title(
    "🛡️ STREESHIELD"
)

st.caption(
    "AI Assistant Test"
)


render_chat()