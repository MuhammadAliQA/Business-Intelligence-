import streamlit as st
import time
import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────
st.set_page_config(
    page_title="ASSBI Platform",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────
from database import init_db, insert_detection, insert_event, clear_all
init_db()

# ─────────────────────────────────────────
# CORE IMPORTS
# ─────────────────────────────────────────
from detector import detect_frame
from stream_capture import capture_burst
from fine_tune import (
    run_fine_tuning,
    save_uploaded_dataset,
    count_dataset_images,
    get_best_model_path
)

from analytics import (
    get_kpis,
    chart_person_over_time,
    chart_anomaly_timeline,
    chart_detection_pie,
    chart_heatmap_hourly,
    chart_confidence_histogram,
    chart_fine_tune_loss,
)

from Chatbot import chat, QUICK_QUESTIONS

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("🎯 ASSBI Platform")

    page = st.radio("Navigation", [
        "Dashboard",
        "Live Detection",
        "Dataset Builder",
        "Analytics",
        "Chatbot",
        "Fine-tuning",
        "Settings"
    ])

    groq_key = st.text_input("Groq API Key", type="password")

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
if page == "Dashboard":

    st.title("📊 Smart Surveillance Dashboard")

    if st.button("⚡ Load Demo Data"):
        generate_demo_data()
        st.success("Demo data loaded!")
        st.rerun()

    k = get_kpis()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Persons", k["current_persons"])
    c2.metric("🚨 Anomaly", str(k["current_anomaly"]))
    c3.metric("📷 Frames", k["total_frames_analyzed"])
    c4.metric("🚗 Vehicles", k["total_vehicles"])

    st.divider()

    st.plotly_chart(chart_person_over_time(), use_container_width=True)
    st.plotly_chart(chart_anomaly_timeline(), use_container_width=True)
    st.plotly_chart(chart_detection_pie(), use_container_width=True)

# ─────────────────────────────────────────
# LIVE DETECTION
# ─────────────────────────────────────────
elif page == "Live Detection":

    st.title("📹 Live Detection")

    img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img:

        # FIX 1: PIL -> numpy (YOLO FIX)
        frame = np.array(Image.open(img).convert("RGB"))

        ann, p, c, o, anomaly, conf = detect_frame(frame)

        insert_detection(p, c, o, anomaly, conf, "upload")

        st.image(frame, caption="Input Image")
        st.image(ann, caption="Detection Result")

        st.write(f"👤 Persons: {p}")
        st.write(f"🚗 Cars: {c}")
        st.write(f"📦 Other: {o}")
        st.write(f"🎯 Confidence: {conf:.2f}")

        if anomaly:
            insert_event("ALERT", "Crowd anomaly detected", "warning")
            st.error("⚠️ CROWD ANOMALY")

# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────
elif page == "Dataset Builder":

    st.title("📸 Dataset Builder")

    url = st.text_input("YouTube URL")
    n_frames = st.slider("Frames", 1, 10, 5)

    if st.button("Capture Frames"):

        frames = capture_burst(url, n_frames=n_frames)

        for f in frames:
            # FIX 2: BGR -> RGB
            st.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

        st.success("Frames captured")

# ─────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────
elif page == "Analytics":

    st.title("📊 Analytics")

    k = get_kpis()

    col1, col2 = st.columns(2)
    col1.metric("Frames", k["total_frames_analyzed"])
    col2.metric("Anomalies", k["total_anomalies"])

    st.plotly_chart(chart_person_over_time(), use_container_width=True)
    st.plotly_chart(chart_heatmap_hourly(), use_container_width=True)
    st.plotly_chart(chart_confidence_histogram(), use_container_width=True)

# ─────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────
elif page == "Chatbot":

    st.title("🤖 AI Chatbot")

    # FIX 3: API safety
    if not groq_key:
        st.warning("Please enter Groq API Key")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    for q in QUICK_QUESTIONS:
        if st.button(q):
            st.session_state.history.append({"role": "user", "content": q})
            reply = chat(st.session_state.history, groq_key)
            st.session_state.history.append({"role": "assistant", "content": reply})

    for m in st.session_state.history:
        st.write(f"**{m['role']}**: {m['content']}")

    msg = st.text_input("Ask anything")

    if st.button("Send") and msg:
        st.session_state.history.append({"role": "user", "content": msg})
        reply = chat(st.session_state.history, groq_key)
        st.session_state.history.append({"role": "assistant", "content": reply})

# ─────────────────────────────────────────
# FINE TUNING
# ─────────────────────────────────────────
elif page == "Fine-tuning":

    st.title("🔧 YOLO Fine-tuning")

    counts = count_dataset_images()
    st.write("Dataset:", counts)

    zip_file = st.file_uploader("Upload Dataset ZIP")

    if zip_file:
        save_uploaded_dataset(zip_file.read())
        st.success("Dataset uploaded")

    epochs = st.slider("Epochs", 5, 50, 10)

    if st.button("Train Model"):
        model_path = run_fine_tuning(
            epochs=epochs,
            batch=8,
            imgsz=640,
            base_model="yolov8n.pt"
        )
        st.success(f"Model saved: {model_path}")

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
elif page == "Settings":

    st.title("⚙️ Settings")

    if st.button("Clear Database"):
        clear_all()
        st.success("Database cleared")

    model_path = get_best_model_path()
    st.info(f"Current model: {model_path}")