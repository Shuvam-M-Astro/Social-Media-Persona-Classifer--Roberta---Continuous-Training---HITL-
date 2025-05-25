import streamlit as st
import sys
import logging
import os

# Allow imports from src/
sys.path.append(os.path.abspath("src"))

# Patch torch for Streamlit on Windows if needed
sys.modules['torch.classes'] = None

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page layout
st.set_page_config(
    page_title="Decode Your Digital Persona",
    layout="centered"
)

# Import page logic
from src.ui.pages import (
    page_form,
    page_confirm,
    page_result,
    page_feedback
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "form"
if "result" not in st.session_state:
    st.session_state.result = {}
if "show_retrain" not in st.session_state:
    st.session_state.show_retrain = False

# Page routing
if st.session_state.page == "form":
    page_form()
elif st.session_state.page == "confirm":
    page_confirm()
elif st.session_state.page == "result":
    page_result()
elif st.session_state.page == "feedback":
    page_feedback()
else:
    st.error("Unknown page state. Please restart the app.")