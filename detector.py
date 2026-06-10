import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os, time
from database import insert_detection, insert_event

MODEL_PATH = "yolov8n.pt"  # auto-downloads on first run
FRAME_DIR = "frames"
ANOMALY_THRESHOLD = 8  # persons > this = anomaly (crowd)

os.makedirs(FRAME_DIR, exist_ok=True)

_model = None

def get_model(custom_path=None):
    global _model
    path = custom_path or MODEL_PATH
    if _model is None:
        _model = YOLO(path)
    return _model

def reset_model():
    global _model
    _model = None

def detect_frame(frame_bgr, save_frame=False, custom_model=None):
    """
    Run YOLOv8 detection on a BGR frame.
    Returns: annotated_frame, person_count, car_count, other_count, anomaly, avg_conf
    """
    model = get_model(custom_model)
    results = model(frame_bgr, verbose=False)[0]

    person_count = 0
    car_count = 0
    other_count = 0
    confidences = []

    annotated = frame_bgr.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidences.append(conf)

        if label == "person":
            person_count += 1
            color = (0, 255, 0)
        elif label in ("car", "truck", "bus", "motorcycle"):
            car_count += 1
            color = (255, 165, 0)
        else:
            other_count += 1
            color = (200, 200, 200)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Crowd density overlay
    h, w = annotated.shape[:2]
    overlay_text = f"Persons: {person_count}  Cars: {car_count}"
    cv2.rectangle(annotated, (0, 0), (300, 32), (0, 0, 0), -1)
    cv2.putText(annotated, overlay_text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    anomaly = person_count >= ANOMALY_THRESHOLD
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    frame_path = None
    if save_frame and anomaly:
        fname = f"{FRAME_DIR}/anomaly_{int(time.time())}.jpg"
        cv2.imwrite(fname, annotated)
        frame_path = fname

    return annotated, person_count, car_count, other_count, anomaly, avg_conf


def capture_youtube_frame(url="https://www.youtube.com/watch?v=M3EYAY2MftI"):
    """
    Grab a single frame from a YouTube live stream via yt-dlp.
    Returns BGR numpy array or None.
    """
    import subprocess, tempfile
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=480]", "-g", url],
            capture_output=True, text=True, timeout=15
        )
        stream_url = result.stdout.strip().split("\n")[0]
        if not stream_url:
            return None

        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception as e:
        print(f"[capture_youtube_frame] {e}")
        return None


def process_video_file(video_path, max_frames=200, save_interval=5):
    """
    Process an uploaded video file frame by frame.
    Saves detections to DB. Returns list of result dicts.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % save_interval == 0:
            ann, pc, cc, oc, anomaly, conf = detect_frame(frame, save_frame=True)
            insert_detection(pc, cc, oc, anomaly, conf, source="upload")
            if anomaly:
                insert_event("CROWD_ANOMALY", f"Frame {frame_idx}: {pc} persons detected", "warning")
            results.append({"frame": frame_idx, "persons": pc, "cars": cc, "anomaly": anomaly})
        frame_idx += 1
        if frame_idx >= max_frames * save_interval:
            break

    cap.release()
    return results


def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))