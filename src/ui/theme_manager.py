import base64
import streamlit as st
import logging
import os

logger = logging.getLogger(__name__)

def set_form_background_video(video_path=os.path.join("assets", "background.mp4")):
    try:
        with open(video_path, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{ background: transparent; }}
            #video-bg {{
                position: fixed;
                top: 0;
                left: 0;
                min-width: 100vw;
                min-height: 100vh;
                z-index: -1;
                object-fit: cover;
            }}
            .video-container {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
            }}
            </style>
            <div class="video-container">
                <video id="video-bg" autoplay loop muted playsinline>
                    <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                </video>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Failed to load video background: {e}")
        st.error("⚠️ Failed to load video background.")

def set_result_background(image_path=os.path.join("assets", "background.jpg")):
    try:
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Failed to load background image: {e}")
        st.error("⚠️ Failed to load background image.")