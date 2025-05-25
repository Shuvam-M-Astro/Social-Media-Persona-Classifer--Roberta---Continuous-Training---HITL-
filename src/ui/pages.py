import streamlit as st
from src.ui.backgrounds import set_form_background_video, set_result_background
from src.logic.predict import predict_persona
from src.logic.feedback import save_feedback, retrain_model
from src.logic.predict import load_classifier  # import at the top

IMAGE_MAP = {
    "Tech Enthusiast": "assets/tech.jpg",
    "Foodie Explorer": "assets/food.jpg",
    "Fitness Buff": "assets/fitness.jpg",
    "Fashion Aficionado": "assets/fashion.jpg",
    "Meme Lord": "assets/meme.jpg"
}

def page_form():
    set_form_background_video()
    st.markdown("<h1 style='text-align: center; color: white;'>What do new visitors think about your profile?</h1>", unsafe_allow_html=True)

    with st.form("persona_form"):
        bio = st.text_area("📄 Enter your short bio:", height=100)
        posts = st.text_area("✍️ Paste 3–5 sample posts or captions:", height=200)
        submitted = st.form_submit_button("🔮 Predict Persona")

        if submitted:
            combined = (bio.strip() + " " + posts.strip()).strip()
            if len(combined.split()) < 5:
                st.error("🚫 Please provide at least 5 meaningful words to analyze.")
            else:
                top_label, scores = predict_persona(combined)
                st.session_state.result = {
                    "bio": bio,
                    "posts": posts,
                    "top_label": top_label,
                    "scores": scores
                }
                st.session_state.page = "confirm"
                st.rerun()


def page_confirm():
    set_result_background()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Would you like to help us improve this app?</h2>", unsafe_allow_html=True)

    if st.button("✅ Yes, I'd like to help"):
        st.session_state.page = "feedback"
        st.session_state.show_retrain = True
        st.rerun()
    elif st.button("🚫 No, just show my result"):
        st.session_state.page = "result"
        st.rerun()


def render_result_card():
    res = st.session_state.result
    col1, col2 = st.columns([2, 1])
    with col1:
        img_path = IMAGE_MAP.get(res["top_label"], "assets/default.jpg")
        st.image(img_path, use_container_width=True)
        st.markdown(
            f"<h2 style='color: white;'>🧠 You are a <span style='color: #FFD700;'>{res['top_label']}</span></h2>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("### 🔍 Other Predictions", unsafe_allow_html=True)
        for label, score in sorted(res["scores"].items(), key=lambda x: x[1], reverse=True):
            if label != res["top_label"]:
                st.markdown(f"- **{label}**: `{score:.4f}`")


def page_result():
    set_result_background()
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_result_card()
    if st.button("🔁 Try Again"):
        st.session_state.page = "form"
        st.rerun()


def page_feedback():
    set_result_background()
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_result_card()

    st.markdown("---")
    st.markdown("### 🧪 Was this prediction correct?")
    feedback = st.radio("Select one:", ["✅ Yes", "❌ No"])

    if feedback == "✅ Yes":
        st.success("Thanks for participating!")

    elif feedback == "❌ No":
        corrected_label = st.text_input("📝 What should the correct label be?")
        if st.button("📩 Submit Feedback"):
            success = save_feedback(
                bio=st.session_state.result["bio"],
                posts=st.session_state.result["posts"],
                corrected_label=corrected_label
            )
            if success:
                st.success("✅ Feedback saved. Thank you!")

    if st.session_state.get("show_retrain"):
        if st.button("🧠 Retrain Model"):
            with st.spinner("Retraining in progress..."):
                success, log = retrain_model()
                if success:
                    st.success("✅ Model retrained successfully!")
                    load_classifier()  # ← reload updated model into pipeline
                else:
                    st.error("❌ Retraining failed.")
                st.text_area("Log Output", log, height=300)

    # ✅ Always show this regardless of retraining
    if st.button("🔁 Take It Again"):
        st.session_state.page = "form"
        st.session_state.show_retrain = False
        st.rerun()