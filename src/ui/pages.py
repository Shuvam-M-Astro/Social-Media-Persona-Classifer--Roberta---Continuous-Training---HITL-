import streamlit as st
from src.ui.backgrounds import set_form_background_video, set_result_background
from src.logic.predict import predict_persona
from src.logic.feedback import save_feedback, retrain_model
from src.logic.predict import load_classifier  # import at the top
import json
import os

def load_id2label():
    """Load the id2label dictionary from the JSON file located in the parent directory's 'logic' subfolder."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logic', 'id2label.json')
    with open(file_path, 'r') as f:
        return json.load(f)

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

    # Handle feedback response
    if feedback == "✅ Yes":
        st.success("Thanks for participating!")

    elif feedback == "❌ No":
        corrected_label = st.text_input("📝 What should the correct label be?")
        
        # Only check after the label is entered
        if corrected_label:
            id2label = load_id2label()  # Load the existing labels

            # Check if the corrected_label exists as a value in the id2label JSON
            if corrected_label not in id2label.values():
                st.warning(f"The label `{corrected_label}` is not recognized. To improve the model, please provide new post headlines.")
                
                new_posts = st.text_area("📝 Provide any new post headlines (optional, comma-separated):")
                
                if new_posts:  # If new posts are entered
                    new_posts_list = [post.strip() for post in new_posts.split(",")]  # Split and clean the posts
                    
                    for post in new_posts_list:
                        if post:  # Only save non-empty posts
                            # Save each post as a new row with the same bio and corrected label
                            success = save_feedback(
                                bio=st.session_state.result["bio"],
                                posts=post,  # Save each new post as a separate row
                                corrected_label=corrected_label
                            )
                            if success:
                                st.success(f"✅ Post '{post}' saved!")
                        else:
                            st.error("❌ Please enter valid post headlines.")
                
                if st.button("📩 Submit Feedback with New Posts"):
                    success = save_feedback(
                        bio=st.session_state.result["bio"],
                        posts=st.session_state.result["posts"],
                        corrected_label=corrected_label
                    )
                    if success:
                        st.success("✅ Feedback saved. Thank you! New posts will help improve the model!")
            
            else:
                if st.button("📩 Submit Feedback"):
                    success = save_feedback(
                        bio=st.session_state.result["bio"],
                        posts=st.session_state.result["posts"],
                        corrected_label=corrected_label
                    )
                    if success:
                        st.success("✅ Feedback saved. Thank you!")

    # **Always Available**: Add New Class Button
    st.markdown("### ➕ Add a New Class")
    new_class_name = st.text_input("📝 Enter the name of the new class:")
    
    if new_class_name:
        st.markdown("Please provide 5 post headlines for this new class:")
        new_class_posts = []
        for i in range(5):
            post = st.text_input(f"📝 Post headline {i + 1}:")
            new_class_posts.append(post.strip())

        if all(new_class_posts):  # Ensure 5 posts are provided
            # Save the new posts for this class
            for post in new_class_posts:
                if post:  # Only save non-empty posts
                    success = save_feedback(
                        bio=st.session_state.result["bio"],
                        posts=post,
                        corrected_label=new_class_name  # New class label
                    )
                    if success:
                        st.success(f"✅ Post for '{new_class_name}' saved!")

            # Add the new class to id2label.json
            id2label = load_id2label()  # Reload to ensure we have the latest version
            id2label[new_class_name] = str(len(id2label))  # Assign the next available ID
            with open(os.path.join(os.path.dirname(__file__), '..', 'logic', 'id2label.json'), 'w') as f:
                json.dump(id2label, f, indent=4)
            st.success(f"✅ New class '{new_class_name}' added successfully!")
        else:
            st.error("❌ Please provide exactly 5 post headlines.")

    # Retraining model logic
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

    # Always show the option to take it again, regardless of retraining
    if st.button("🔁 Take It Again"):
        st.session_state.page = "form"
        st.session_state.show_retrain = False
        st.rerun()
