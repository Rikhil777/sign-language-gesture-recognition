import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ================= CONFIG =================
st.set_page_config(page_title="AI Sign", layout="wide")

# ================= SESSION =================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================= THEME =================
def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ================= THEME COLORS =================
if st.session_state.theme == "dark":
    bg_gradient = "linear-gradient(-45deg, #0f2027, #203a43, #2c5364)"
    card = "rgba(255,255,255,0.08)"
    text = "white"
else:
    bg_gradient = "linear-gradient(-45deg, #e0eafc, #cfdef3)"
    card = "rgba(255,255,255,0.9)"
    text = "black"

# ================= MODERN STYLING =================
st.markdown(f"""
<style>
.stApp {{
    background: {bg_gradient};
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: {text};
}}

@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

header, #MainMenu, footer {{
    visibility: hidden;
}}

.main-title {{
    font-size: 52px;
    font-weight: 800;
    text-align:center;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease-in-out;
}}

.glass {{
    background: {card};
    backdrop-filter: blur(20px);
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    animation: fadeIn 1.2s ease-in-out;
    margin-top: 30px;
}}

@keyframes fadeIn {{
    from {{opacity: 0; transform: translateY(20px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}

.stButton>button {{
    width:100%;
    border-radius:15px;
    padding:12px;
    font-weight:600;
    background: linear-gradient(90deg, #00dbde, #fc00ff);
    color:white;
    border:none;
    transition:0.3s;
}}

.stButton>button:hover {{
    transform: scale(1.05);
    box-shadow:0px 0px 20px rgba(0,255,255,0.6);
}}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🤖 AI Sign")

    if st.button("🏠 Home"):
        st.session_state.page = "Home"

    if st.button("📷 Camera"):
        st.session_state.page = "Camera"

    if st.button("📚 Alphabets"):
        st.session_state.page = "Alphabets"

    st.markdown("---")
    st.button("🌗 Toggle Theme", on_click=toggle_theme)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= ROUTER =================
page = st.session_state.page

# ================= HOME =================
if page == "Home":

    st.markdown("<div class='main-title'>AI Sign Language Recognition</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass'>
    <h3>🚀 About This Platform</h3>
    <p>
    AI Sign is an intelligent real-time sign language recognition system.
    It detects hand gestures using deep learning-based landmark detection
    and classifies them using a trained machine learning model.
    </p>

    <h3>🧠 Technologies Used</h3>
    <p>
    • MediaPipe (Hand Landmark Detection)<br>
    • Machine Learning Classifier<br>
    • WebRTC Real-Time Streaming<br>
    • Streamlit Interactive UI
    </p>

    <h3>🎯 Use Cases</h3>
    <p>
    • Assistive communication tool<br>
    • Sign language learning support<br>
    • AI-powered educational demo<br>
    • Real-time gesture recognition system
    </p>
    </div>
    """, unsafe_allow_html=True)

# ================= FAST CAMERA =================

elif page == "Camera":

    st.markdown("<div class='main-title'>Live AI Detection</div>", unsafe_allow_html=True)

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = ""

    # ================= START CAMERA BUTTON =================
    if not st.session_state.camera_active:
        if st.button("▶ Start Camera"):
            st.session_state.camera_active = True
            st.rerun()

    # ================= WHEN CAMERA ACTIVE =================
    if st.session_state.camera_active:

        col1, col2 = st.columns([1,1])

        with col1:
            if st.button("⏹ Stop Camera"):
                st.session_state.camera_active = False
                st.session_state.current_prediction = ""
                st.rerun()

        with col2:
            st.write("")  # Keeps layout aligned (device selector will appear here)

        # ===== Transformer =====
        class SignTransformer(VideoTransformerBase):
            def __init__(self):
                self.hands = mp_hands.Hands(max_num_hands=1)

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    features = []

                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])

                    features = np.array(features).reshape(1, -1)
                    prediction = model.predict(features)[0]

                    st.session_state.current_prediction = prediction

                    mp_draw.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                return img

        # ===== AUTO START CAMERA (NO RED BUTTON) =====
        webrtc_ctx = webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=SignTransformer,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

        # ===== DETECTION BOX BELOW CAMERA =====
        if st.session_state.current_prediction != "":
            st.markdown(
                f"""
                <div style="
                    margin-top:30px;
                    padding:30px;
                    border-radius:20px;
                    background: linear-gradient(90deg,#00dbde,#fc00ff);
                    color:white;
                    text-align:center;
                    font-size:40px;
                    font-weight:bold;
                    box-shadow:0px 0px 30px rgba(0,255,255,0.7);
                ">
                    Detected Sign: {st.session_state.current_prediction}
                </div>
                """,
                unsafe_allow_html=True
            )
            
# ================= ALPHABETS =================
elif page == "Alphabets":

    st.markdown("<div class='main-title'>Sign Language Alphabets</div>", unsafe_allow_html=True)

    image_dir = os.path.join(os.path.dirname(__file__), "images")

    if os.path.exists(image_dir):
        images = sorted(os.listdir(image_dir))
        cols = st.columns(6)

        for i, img in enumerate(images):
            img_path = os.path.join(image_dir, img)
            cols[i % 6].image(img_path, caption=img.split(".")[0], use_column_width=True)
    else:
        st.warning("No images folder found inside frontend/")