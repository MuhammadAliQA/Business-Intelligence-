import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASSBI Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric {
    background: #1a1d27;
    padding: 12px;
    border-radius: 10px;
    border-left: 3px solid #00C896;
}
.stMetric label { color: #aaa !important; font-size: 13px !important; }
.alert-box {
    background: #2d1b1b;
    border: 1px solid #FF4B6E;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #FF4B6E;
    font-size: 13px;
}
.info-box {
    background: #1b2d1b;
    border: 1px solid #00C896;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #00C896;
    font-size: 13px;
}
.chat-user {
    background: #1e2740;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #e0e7ff;
}
.chat-bot {
    background: #1a2a1a;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #d0ffe0;
}
div[data-testid="stSidebar"] { background: #13151f; }
h1, h2, h3 { color: #e8e8e8 !important; }
.dataset-card {
    background: #1a1d27;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #2a2d3a;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Init DB ────────────────────────────────────────────────────────────────────
from database import init_db
init_db()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-camera.png", width=60)
    st.title("ASSBI Platform")
    st.caption("AI Smart Surveillance & BI")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📹 Live Detection",
        "📸 Dataset Builder",
        "📊 Analytics",
        "🤖 AI Chatbot",
        "🔧 Fine-tuning",
        "⚙️ Settings"
    ])

    st.divider()
    st.markdown("**API Keys**")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key at console.groq.com",
        key="groq_key"
    )
    st.divider()
    st.caption("BTEC Unit 12 · Business Intelligence")
    st.caption("ASSBI Platform v2.0")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🎯 ASSBI — Smart Surveillance Dashboard")
    st.caption("AI-Powered Business Intelligence | Real-time monitoring of Abbey Road, London")

    from Analytics import get_kpis, chart_person_over_time, chart_anomaly_timeline, generate_demo_data
    from database import get_events

    col_btn1, col_btn2 = st.columns([1, 6])
    with col_btn1:
        if st.button("⚡ Load Demo Data"):
            generate_demo_data()
            st.success("120 demo records loaded!")
            st.rerun()

    kpis = get_kpis()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("👤 Current Persons", kpis["current_persons"])
    c2.metric(
        "🚨 Anomaly",
        "YES" if kpis["current_anomaly"] else "NO",
        delta="⚠️ Alert" if kpis["current_anomaly"] else None,
        delta_color="inverse"
    )
    c3.metric("📷 Frames Analyzed", kpis["total_frames_analyzed"])
    c4.metric("🔴 Total Anomalies", kpis["total_anomalies"])
    c5.metric("📊 Avg Persons/Frame", kpis["avg_persons"])
    c6.metric("🚗 Total Vehicles", kpis["total_vehicles"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Person & Vehicle Trend")
        st.plotly_chart(chart_person_over_time(), use_container_width=True)
    with col2:
        st.subheader("Anomaly Timeline")
        st.plotly_chart(chart_anomaly_timeline(), use_container_width=True)

    st.subheader("Recent Events")
    events_df = get_events(10)
    if not events_df.empty:
        for _, row in events_df.iterrows():
            box_class = "alert-box" if row["severity"] == "warning" else "info-box"
            icon = "⚠️" if row["severity"] == "warning" else "ℹ️"
            st.markdown(
                f'<div class="{box_class}">'
                f'{icon} <b>[{row["timestamp"]}]</b> {row["event_type"]}: {row["description"]}'
                f'</div>', unsafe_allow_html=True
            )
    else:
        st.info("No events yet. Run Live Detection or load demo data.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📹 Live Detection":
    st.title("📹 Live Object Detection")
    st.caption("YOLOv8 · OpenCV · Real-time surveillance of Abbey Road")

    from detector import detect_frame, capture_youtube_frame, process_video_file, bgr_to_pil, pil_to_bgr
    from database import insert_detection, insert_event
    from fine_tune import get_best_model_path

    mode = st.radio("Detection Mode", [
        "📸 YouTube Live Frame",
        "⬆️ Upload Image",
        "🎬 Upload Video File"
    ], horizontal=True)

    custom_model = get_best_model_path()
    if custom_model:
        use_custom = st.checkbox(f"Use fine-tuned model: `{custom_model}`", value=True)
        model_path = custom_model if use_custom else None
    else:
        model_path = None
        st.info("Using YOLOv8n (pretrained) — upload a dataset and fine-tune for better accuracy.")

    st.divider()

    # ── YouTube Live Frame ──
    if mode == "📸 YouTube Live Frame":
        yt_url = st.text_input(
            "YouTube URL",
            value="https://www.youtube.com/watch?v=M3EYAY2MftI",
            help="Abbey Road live cam or any YouTube live stream"
        )

        col_run, col_auto = st.columns([1, 2])
        with col_run:
            run_once = st.button("📸 Capture & Detect", type="primary")
        with col_auto:
            auto_refresh = st.checkbox("🔄 Auto-refresh every 5 seconds")

        if run_once or auto_refresh:
            with st.spinner("Fetching frame from YouTube live stream..."):
                frame = capture_youtube_frame(yt_url)

            if frame is None:
                st.error("Could not capture frame. Check URL, internet connection, or that the stream is live.")
                st.info("💡 Try 'Upload Image' mode to test detection offline.")
            else:
                ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=True, custom_model=model_path)
                insert_detection(pc, cc, oc, anomaly, conf, source="youtube")
                if anomaly:
                    insert_event("CROWD_ANOMALY", f"{pc} persons detected at Abbey Road", "warning")

                col_img, col_stats = st.columns([3, 1])
                with col_img:
                    st.image(bgr_to_pil(ann), caption="YOLOv8 Detection Result", use_column_width=True)
                with col_stats:
                    st.metric("👤 Persons", pc)
                    st.metric("🚗 Vehicles", cc)
                    st.metric("📦 Other Objects", oc)
                    st.metric("🎯 Avg Confidence", f"{conf:.2%}")
                    if anomaly:
                        st.markdown('<div class="alert-box">⚠️ CROWD ANOMALY<br>≥8 persons detected</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">✅ Normal Activity</div>',
                                    unsafe_allow_html=True)

            if auto_refresh:
                time.sleep(5)
                st.rerun()

    # ── Upload Image ──
    elif mode == "⬆️ Upload Image":
        uploaded = st.file_uploader("Upload image for detection", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            frame = pil_to_bgr(pil_img)

            with st.spinner("Running YOLOv8 detection..."):
                ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=False, custom_model=model_path)
                insert_detection(pc, cc, oc, anomaly, conf, source="image_upload")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(bgr_to_pil(ann), caption="Detection Result", use_column_width=True)
            with col2:
                st.metric("👤 Persons", pc)
                st.metric("🚗 Vehicles", cc)
                st.metric("📦 Other", oc)
                st.metric("🎯 Confidence", f"{conf:.2%}")
                if anomaly:
                    st.markdown('<div class="alert-box">⚠️ CROWD ANOMALY</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">✅ Normal</div>', unsafe_allow_html=True)

    # ── Upload Video ──
    elif mode == "🎬 Upload Video File":
        uploaded_vid = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        max_frames = st.slider("Max frames to analyze", 50, 500, 150, 50)

        if uploaded_vid and st.button("▶️ Start Video Analysis", type="primary"):
            tmp_path = f"/tmp/upload_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_vid.read())

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("Analyzing video frames..."):
                results = process_video_file(tmp_path, max_frames=max_frames)
                progress.progress(100)

            if results:
                import pandas as pd
                status.success(f"✅ Analysis complete! {len(results)} frames processed.")
                total_anomalies = sum(1 for r in results if r["anomaly"])
                c1, c2 = st.columns(2)
                c1.metric("Frames Processed", len(results))
                c2.metric("Anomaly Frames", total_anomalies)

                df = pd.DataFrame(results)
                st.line_chart(df.set_index("frame")[["persons", "cars"]])
            else:
                st.warning("No frames processed. Check video file.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📸 Dataset Builder":
    st.title("📸 Dataset Builder")
    st.caption("Capture screenshots from Abbey Road live stream → build person detection dataset")

    from detector import capture_youtube_frame, detect_frame, bgr_to_pil
    from stream_capture import capture_burst, save_frame_for_dataset
    from database import insert_event
    import os

    st.info("""
    **How it works:**
    1. Click **Capture Frames** to grab screenshots from the Abbey Road live cam
    2. YOLOv8 detects persons in each frame
    3. Frames are saved to `dataset/images/` for fine-tuning
    4. Go to **Fine-tuning** page to train a custom model on your captured data
    """)

    yt_url = st.text_input(
        "Stream URL",
        value="https://www.youtube.com/watch?v=M3EYAY2MftI",
        help="Abbey Road live cam"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_frames = st.slider("Frames to capture", 1, 20, 5)
    with col2:
        interval = st.slider("Interval (seconds)", 1, 10, 3)
    with col3:
        split = st.radio("Dataset split", ["train", "val"], horizontal=True)

    if st.button("📸 Start Capturing", type="primary"):
        saved_paths = []
        progress = st.progress(0)
        preview_cols = st.columns(min(n_frames, 5))

        for i in range(n_frames):
            with st.spinner(f"Capturing frame {i+1}/{n_frames}..."):
                frame = capture_youtube_frame(yt_url)

            if frame is not None:
                path = save_frame_for_dataset(frame, split=split)
                saved_paths.append(path)

                # Show annotated preview
                ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=False)
                with preview_cols[i % 5]:
                    st.image(bgr_to_pil(ann), caption=f"Frame {i+1}: {pc}P {cc}C",
                             use_column_width=True)

                insert_event(
                    "DATASET_CAPTURE",
                    f"Frame {i+1} captured → {path} | Persons: {pc}",
                    "info"
                )

            progress.progress((i + 1) / n_frames)

            if i < n_frames - 1:
                time.sleep(interval)

        st.success(f"✅ {len(saved_paths)}/{n_frames} frames saved to `dataset/images/{split}/`")

    # Show current dataset stats
    st.divider()
    st.subheader("Current Dataset")

    from fine_tune import count_dataset_images
    counts = count_dataset_images()

    col_a, col_b = st.columns(2)
    col_a.metric("🏋️ Train Images", counts["train"])
    col_b.metric("🧪 Val Images", counts["val"])

    # Show saved frames
    frames_dir = "dataset/images"
    if counts["train"] > 0 or counts["val"] > 0:
        show_split = st.selectbox("Preview", ["train", "val"])
        split_dir = f"{frames_dir}/{show_split}"
        if os.path.exists(split_dir):
            files = [f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png"))][:12]
            if files:
                cols = st.columns(4)
                for idx, fname in enumerate(files):
                    fpath = os.path.join(split_dir, fname)
                    try:
                        img = Image.open(fpath)
                        with cols[idx % 4]:
                            st.image(img, caption=fname, use_column_width=True)
                    except Exception:
                        pass
    else:
        st.info("No dataset images yet. Use the capture tool above.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Business Intelligence Analytics")
    st.caption("KPIs · Trend Charts · Heatmaps · Confidence Distribution")

    from Analytics import (
        chart_person_over_time, chart_anomaly_timeline,
        chart_detection_pie, chart_heatmap_hourly,
        chart_confidence_histogram, get_kpis
    )
    from database import get_detections
    import pandas as pd

    kpis = get_kpis()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Frames", kpis["total_frames_analyzed"])
    c2.metric("Total Anomalies", kpis["total_anomalies"])
    c3.metric("Avg Persons/Frame", kpis["avg_persons"])
    c4.metric("Total Vehicles", kpis["total_vehicles"])

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend", "🔴 Anomalies", "🥧 Distribution", "🌡️ Heatmap", "📉 Confidence"
    ])

    with tab1:
        st.plotly_chart(chart_person_over_time(), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_anomaly_timeline(), use_container_width=True)
    with tab3:
        st.plotly_chart(chart_detection_pie(), use_container_width=True)
    with tab4:
        st.plotly_chart(chart_heatmap_hourly(), use_container_width=True)
    with tab5:
        st.plotly_chart(chart_confidence_histogram(), use_container_width=True)

    st.divider()
    st.subheader("📋 Raw Detection Data")
    df = get_detections(300)
    if not df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_anomalies_only = st.checkbox("Show anomalies only")
        with col_f2:
            min_persons = st.slider("Min persons filter", 0, 20, 0)

        filtered = df.copy()
        if show_anomalies_only:
            filtered = filtered[filtered["anomaly"] == 1]
        filtered = filtered[filtered["person_count"] >= min_persons]

        st.dataframe(filtered, use_container_width=True, height=320)
        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "detections.csv", "text/csv")
    else:
        st.info("No data yet. Run detection or load demo data from Dashboard.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.title("🤖 ASSBI AI Assistant")
    st.caption("Powered by Groq · LLaMA 3.3 70B · Context-aware surveillance analytics chatbot")

    from Chatbot import chat, QUICK_QUESTIONS

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not groq_key:
        st.warning("⚠️ Enter your **Groq API key** in the sidebar to activate the chatbot.")
        st.info("Get a **free** API key at [console.groq.com](https://console.groq.com) — no credit card needed.")
        st.stop()

    # Quick question buttons
    st.subheader("Quick Questions")
    q_cols = st.columns(4)
    for i, q in enumerate(QUICK_QUESTIONS):
        with q_cols[i % 4]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("ASSBI AI is thinking..."):
                    reply = chat(st.session_state.chat_history, groq_key)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.divider()

    # Chat history display
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 <b>ASSBI AI:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown('<div class="info-box">👋 Ask me anything about the surveillance data, anomalies, trends, or how the system works!</div>',
                    unsafe_allow_html=True)

    st.divider()

    # Input row
    col_input, col_send, col_clear = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            label_visibility="collapsed",
            placeholder="Ask about persons detected, anomalies, trends, accuracy..."
        )
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            reply = chat(st.session_state.chat_history, groq_key)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Fine-tuning":
    st.title("🔧 YOLOv8 Fine-tuning")
    st.caption("Train a custom person detection model on your captured Abbey Road dataset")

    from fine_tune import (
        run_fine_tuning, save_uploaded_dataset,
        count_dataset_images, get_best_model_path
    )
    from Analytics import chart_fine_tune_loss

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset")
        counts = count_dataset_images()

        if counts["train"] > 0:
            st.success(f"✅ Dataset ready — Train: **{counts['train']}** · Val: **{counts['val']}** images")
        else:
            st.info("No dataset yet. Use the **Dataset Builder** page to capture frames, or upload a ZIP.")

        st.markdown("**Or upload a ZIP dataset:**")
        st.markdown("""
```
dataset.zip/
├── images/train/*.jpg
├── images/val/*.jpg
├── labels/train/*.txt   ← YOLO format: class cx cy w h
└── labels/val/*.txt
```
""")
        dataset_zip = st.file_uploader("Upload dataset ZIP", type=["zip"])
        if dataset_zip:
            if st.button("📦 Extract & Prepare"):
                with st.spinner("Extracting dataset..."):
                    save_uploaded_dataset(dataset_zip.read())
                counts = count_dataset_images()
                st.success(f"Dataset ready — Train: {counts['train']} · Val: {counts['val']}")

    with col2:
        st.subheader("Training Config")
        epochs = st.slider("Epochs", 5, 100, 20)
        batch = st.select_slider("Batch size", [4, 8, 16, 32], value=8)
        imgsz = st.select_slider("Image size", [320, 416, 512, 640], value=640)
        base_model = st.selectbox(
            "Base model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            help="n = fastest · m = most accurate"
        )

        st.divider()
        best = get_best_model_path()
        if best:
            st.success(f"✅ Fine-tuned model: `{best}`")
        else:
            st.info("No fine-tuned model yet.")

    st.divider()

    if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

        def update_progress(ep, total, loss, acc):
            pct = int(ep / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Epoch {ep}/{total} — Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            chart_placeholder.plotly_chart(chart_fine_tune_loss(), use_container_width=True)

        with st.spinner("Training in progress..."):
            best_path = run_fine_tuning(
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                base_model=base_model,
                progress_callback=update_progress
            )

        progress_bar.progress(100)

        if best_path:
            st.success(f"✅ Fine-tuning complete! Best model: `{best_path}`")
        else:
            st.success("✅ Training simulation complete! Upload a real dataset for actual model training.")

    st.subheader("📈 Training Loss History")
    st.plotly_chart(chart_fine_tune_loss(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    from database import clear_all, get_stats_summary
    from detector import reset_model

    st.subheader("System Stats")
    stats = get_stats_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("DB Records", stats["total_frames"])
    c2.metric("Total Anomalies", stats["total_anomalies"])
    c3.metric("Total Persons Detected", stats["total_persons_detected"])

    st.divider()

    st.subheader("Anomaly Threshold")
    st.info("Currently: **≥8 persons** in a single frame triggers a crowd anomaly alert. Edit `ANOMALY_THRESHOLD` in `detector.py` to adjust.")

    st.divider()

    st.subheader("Data Management")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear All Detection Data", type="secondary"):
            clear_all()
            st.success("All detection data cleared.")
            st.rerun()
    with col_b:
        if st.button("🔄 Reset YOLO Model Cache"):
            reset_model()
            st.success("Model cache reset — will reload on next detection.")

    st.divider()

    st.subheader("About ASSBI Platform")
    st.markdown("""
| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Video Processing | OpenCV |
| Live Stream Capture | yt-dlp |
| Database | SQLite |
| BI Dashboard | Streamlit + Plotly |
| AI Chatbot | Groq API (LLaMA 3.3 70B) |
| Fine-tuning | YOLOv8 custom training |
| Language | Python 3.10+ |

**BTEC Unit 12 — Business Intelligence**  
LO1 · LO2 · LO3 · LO4 covered
""")