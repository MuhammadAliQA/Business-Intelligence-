import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import io
import threading

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
    .stMetric { background: #1a1d27; padding: 12px; border-radius: 10px;
                border-left: 3px solid #00C896; }
    .stMetric label { color: #aaa !important; font-size: 13px !important; }
    .alert-box { background: #2d1b1b; border: 1px solid #FF4B6E;
                 border-radius: 8px; padding: 10px 14px; margin: 6px 0;
                 color: #FF4B6E; font-size: 13px; }
    .info-box  { background: #1b2d1b; border: 1px solid #00C896;
                 border-radius: 8px; padding: 10px 14px; margin: 6px 0;
                 color: #00C896; font-size: 13px; }
    .chat-user { background: #1e2740; border-radius: 10px;
                 padding: 8px 12px; margin: 4px 0; }
    .chat-bot  { background: #1a2a1a; border-radius: 10px;
                 padding: 8px 12px; margin: 4px 0; }
    div[data-testid="stSidebar"] { background: #13151f; }
    h1, h2, h3 { color: #e8e8e8 !important; }
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
        "📊 Analytics",
        "🤖 AI Chatbot",
        "🔧 Fine-tuning",
        "⚙️ Settings"
    ])

    st.divider()
    st.markdown("**API Keys**")
    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get free key at console.groq.com",
                              key="groq_key")
    st.divider()
    st.caption("BTEC Unit 12 · Business Intelligence")
    st.caption("Smart Surveillance Platform v1.0")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🎯 ASSBI — Smart Surveillance Dashboard")
    st.caption("AI-Powered Business Intelligence Platform | Real-time monitoring")

    from analytics import get_kpis, chart_person_over_time, chart_anomaly_timeline, generate_demo_data
    from database import get_events

    # Demo data button
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if st.button("⚡ Load Demo Data"):
            generate_demo_data()
            st.success("120 demo records loaded!")
            st.rerun()

    kpis = get_kpis()

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("👤 Current Persons", kpis["current_persons"])
    c2.metric("🚨 Anomaly", "YES" if kpis["current_anomaly"] else "NO",
              delta="⚠️ Alert" if kpis["current_anomaly"] else None,
              delta_color="inverse")
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
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("No events yet. Run detection or load demo data.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📹 Live Detection":
    st.title("📹 Live Object Detection")
    st.caption("YOLOv8 · OpenCV · Real-time surveillance")

    from detector import detect_frame, capture_youtube_frame, process_video_file, bgr_to_pil, pil_to_bgr
    from database import insert_detection, insert_event
    from fine_tune import get_best_model_path

    mode = st.radio("Detection Mode", [
        "📸 Single YouTube Frame",
        "⬆️ Upload Image",
        "🎬 Upload Video File"
    ], horizontal=True)

    custom_model = get_best_model_path()
    if custom_model:
        use_custom = st.checkbox(f"Use fine-tuned model: `{custom_model}`", value=True)
        model_path = custom_model if use_custom else None
    else:
        model_path = None
        st.info("No fine-tuned model found — using YOLOv8n (pretrained)")

    st.divider()

    # ── Single YouTube Frame ──
    if mode == "📸 Single YouTube Frame":
        yt_url = st.text_input("YouTube URL",
                               value="https://www.youtube.com/watch?v=M3EYAY2MftI",
                               help="Abbey Road live cam")
        col_run, col_live = st.columns([1, 3])
        with col_run:
            run_once = st.button("📸 Capture & Detect", type="primary")
            auto_refresh = st.checkbox("🔄 Auto refresh (5s)")

        if run_once or auto_refresh:
            with st.spinner("Fetching frame from YouTube..."):
                frame = capture_youtube_frame(yt_url)

            if frame is None:
                st.error("Could not fetch frame. Check URL or internet connection.")
                st.info("💡 Tip: Try uploading an image instead for offline testing.")
            else:
                ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=True, custom_model=model_path)
                insert_detection(pc, cc, oc, anomaly, conf, source="youtube")
                if anomaly:
                    insert_event("CROWD_ANOMALY", f"{pc} persons detected at Abbey Road", "warning")

                col_img, col_stats = st.columns([3, 1])
                with col_img:
                    st.image(bgr_to_pil(ann), caption="Detected frame", use_column_width=True)
                with col_stats:
                    st.metric("👤 Persons", pc)
                    st.metric("🚗 Vehicles", cc)
                    st.metric("📦 Other", oc)
                    st.metric("🎯 Avg Confidence", f"{conf:.2%}")
                    if anomaly:
                        st.markdown('<div class="alert-box">⚠️ CROWD ANOMALY DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">✅ Normal activity</div>', unsafe_allow_html=True)

            if auto_refresh:
                time.sleep(5)
                st.rerun()

    # ── Upload Image ──
    elif mode == "⬆️ Upload Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            frame = pil_to_bgr(pil_img)
            ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=False, custom_model=model_path)
            insert_detection(pc, cc, oc, anomaly, conf, source="upload")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(bgr_to_pil(ann), caption="Detection result", use_column_width=True)
            with col2:
                st.metric("👤 Persons", pc)
                st.metric("🚗 Vehicles", cc)
                st.metric("📦 Other", oc)
                st.metric("🎯 Confidence", f"{conf:.2%}")
                if anomaly:
                    st.markdown('<div class="alert-box">⚠️ CROWD ANOMALY</div>', unsafe_allow_html=True)

    # ── Upload Video ──
    elif mode == "🎬 Upload Video File":
        uploaded_vid = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        max_frames = st.slider("Max frames to analyze", 50, 500, 100, 50)

        if uploaded_vid and st.button("▶️ Start Analysis", type="primary"):
            tmp_path = f"/tmp/upload_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_vid.read())

            with st.spinner("Analyzing video..."):
                progress = st.progress(0)
                results = process_video_file(tmp_path, max_frames=max_frames)
                progress.progress(100)

            st.success(f"Analysis complete! {len(results)} frames processed.")
            total_anomalies = sum(1 for r in results if r["anomaly"])
            st.metric("Anomaly frames", total_anomalies)
            import pandas as pd
            df = pd.DataFrame(results)
            st.line_chart(df.set_index("frame")[["persons", "cars"]])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Business Intelligence Analytics")
    st.caption("KPIs · Charts · Heatmaps · Predictive insights")

    from analytics import (chart_person_over_time, chart_anomaly_timeline,
                           chart_detection_pie, chart_heatmap_hourly,
                           chart_confidence_histogram, get_kpis)
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
    df = get_detections(200)
    if not df.empty:
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            show_anomalies_only = st.checkbox("Show anomalies only")
        with col_filter2:
            min_persons = st.slider("Min persons filter", 0, 20, 0)

        filtered = df.copy()
        if show_anomalies_only:
            filtered = filtered[filtered["anomaly"] == 1]
        filtered = filtered[filtered["person_count"] >= min_persons]

        st.dataframe(filtered, use_container_width=True, height=300)

        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "detections_export.csv", "text/csv")
    else:
        st.info("No data yet. Run detection first or load demo data from Dashboard.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.title("🤖 ASSBI AI Assistant")
    st.caption("Powered by Groq · LLaMA 3.3 70B · Context-aware BI chatbot")

    from chatbot import chat, QUICK_QUESTIONS

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not groq_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar. Get a free key at [console.groq.com](https://console.groq.com)")
        st.info("The chatbot uses LLaMA 3.3 70B via Groq's free API. No credit card required.")
        st.stop()

    # Quick questions
    st.subheader("Quick questions")
    cols = st.columns(4)
    for i, q in enumerate(QUICK_QUESTIONS):
        with cols[i % 4]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    reply = chat(st.session_state.chat_history, groq_key)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 <b>ASSBI AI:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Input
    st.divider()
    col_input, col_send, col_clear = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input("Ask anything about the surveillance data...",
                                   label_visibility="collapsed",
                                   placeholder="e.g. How many anomalies were detected?")
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("ASSBI AI is thinking..."):
            reply = chat(st.session_state.chat_history, groq_key)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Fine-tuning":
    st.title("🔧 YOLOv8 Fine-tuning")
    st.caption("Custom model training on person detection dataset")

    from fine_tune import (run_fine_tuning, save_uploaded_dataset,
                           count_dataset_images, get_best_model_path)
    from analytics import chart_fine_tune_loss

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Upload")
        st.markdown("""
        **Expected ZIP structure:**
        ```
        dataset.zip/
        ├── images/train/*.jpg
        ├── images/val/*.jpg
        ├── labels/train/*.txt
        └── labels/val/*.txt
        ```
        Labels are YOLO format: `class cx cy w h`
        """)
        dataset_zip = st.file_uploader("Upload dataset ZIP", type=["zip"])
        if dataset_zip:
            if st.button("📦 Extract Dataset"):
                with st.spinner("Extracting..."):
                    save_uploaded_dataset(dataset_zip.read())
                counts = count_dataset_images()
                st.success(f"Dataset ready — Train: {counts['train']} images, Val: {counts['val']} images")

        counts = count_dataset_images()
        if counts["train"] > 0:
            st.info(f"📁 Current dataset — Train: {counts['train']}, Val: {counts['val']}")

    with col2:
        st.subheader("Training Config")
        epochs = st.slider("Epochs", 5, 100, 20)
        batch = st.select_slider("Batch size", [4, 8, 16, 32], value=8)
        imgsz = st.select_slider("Image size", [320, 416, 512, 640], value=640)
        base_model = st.selectbox("Base model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                                  help="n=fastest, m=most accurate")

        st.markdown("---")
        best = get_best_model_path()
        if best:
            st.success(f"✅ Fine-tuned model available: `{best}`")
        else:
            st.info("No fine-tuned model yet.")

    st.divider()

    if st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        def update_progress(ep, total, loss, acc):
            pct = int(ep / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Epoch {ep}/{total} — Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            metrics_placeholder.plotly_chart(chart_fine_tune_loss(), use_container_width=True)

        with st.spinner("Training in progress..."):
            best_path = run_fine_tuning(
                epochs=epochs, batch=batch, imgsz=imgsz,
                base_model=base_model,
                progress_callback=update_progress
            )

        progress_bar.progress(100)
        if best_path:
            st.success(f"✅ Fine-tuning complete! Best model saved at: `{best_path}`")
        else:
            st.success("✅ Training simulation complete! (Upload a dataset for real training)")

    st.subheader("Training History")
    st.plotly_chart(chart_fine_tune_loss(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")

    from database import clear_all, get_stats_summary
    from detector import reset_model

    st.subheader("System Info")
    stats = get_stats_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("DB Records", stats["total_frames"])
    col2.metric("Total Anomalies", stats["total_anomalies"])
    col3.metric("Total Persons Detected", stats["total_persons_detected"])

    st.divider()

    st.subheader("Anomaly Threshold")
    st.info("Currently: **8 persons** triggers a crowd anomaly alert. Edit `ANOMALY_THRESHOLD` in `detector.py` to change.")

    st.divider()

    st.subheader("Data Management")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear All Detection Data", type="secondary"):
            clear_all()
            st.success("All detection data cleared.")
    with col_b:
        if st.button("🔄 Reset YOLO Model Cache"):
            reset_model()
            st.success("Model cache reset.")

    st.divider()

    st.subheader("About ASSBI Platform")
    st.markdown("""
    | Component | Technology |
    |-----------|-----------|
    | Object Detection | YOLOv8 (Ultralytics) |
    | Video Processing | OpenCV |
    | Video Capture | yt-dlp |
    | Database | SQLite |
    | BI Dashboard | Streamlit + Plotly |
    | AI Chatbot | Groq API (LLaMA 3.3 70B) |
    | Fine-tuning | YOLOv8 custom training |
    | Language | Python 3.10+ |

    **BTEC Unit 12 — Business Intelligence**
    LO1 · LO2 · LO3 · LO4 covered
    """)