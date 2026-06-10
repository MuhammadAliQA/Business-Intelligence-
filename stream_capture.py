"""
stream_capture.py
Utility module for capturing frames from YouTube live streams
using yt-dlp and OpenCV.
"""

import cv2
import subprocess
import time
import os
import numpy as np
from PIL import Image

ABBEY_ROAD_URL = "https://www.youtube.com/watch?v=M3EYAY2MftI"
FRAMES_DIR = "frames"

os.makedirs(FRAMES_DIR, exist_ok=True)


def get_stream_url(youtube_url: str, quality: str = "best[height<=480]") -> str | None:
    """
    Resolve a YouTube URL to a direct stream URL via yt-dlp.
    Returns the stream URL string or None on failure.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", quality, "-g", youtube_url],
            capture_output=True, text=True, timeout=20
        )
        lines = result.stdout.strip().split("\n")
        stream_url = lines[0] if lines else None
        return stream_url if stream_url else None
    except Exception as e:
        print(f"[get_stream_url] Error: {e}")
        return None


def capture_single_frame(youtube_url: str = ABBEY_ROAD_URL) -> np.ndarray | None:
    """
    Capture a single BGR frame from a YouTube live stream.
    Returns numpy BGR array or None on failure.
    """
    stream_url = get_stream_url(youtube_url)
    if not stream_url:
        return None

    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Skip a couple frames to get a clean one
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception as e:
        print(f"[capture_single_frame] Error: {e}")
        return None


def capture_burst(youtube_url: str = ABBEY_ROAD_URL,
                  n_frames: int = 5,
                  interval_sec: float = 2.0,
                  save_to_disk: bool = True) -> list[np.ndarray]:
    """
    Capture n_frames frames from a YouTube live stream with a given interval.
    Optionally saves frames as JPEGs for dataset building.
    Returns list of BGR numpy arrays.
    """
    stream_url = get_stream_url(youtube_url)
    if not stream_url:
        print("[capture_burst] Could not resolve stream URL.")
        return []

    frames = []
    for i in range(n_frames):
        try:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frames.append(frame)
                if save_to_disk:
                    path = os.path.join(FRAMES_DIR, f"capture_{int(time.time())}_{i}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"[capture_burst] Saved: {path}")

        except Exception as e:
            print(f"[capture_burst] Frame {i} error: {e}")

        if i < n_frames - 1:
            time.sleep(interval_sec)

    print(f"[capture_burst] Captured {len(frames)}/{n_frames} frames.")
    return frames


def save_frame_for_dataset(frame_bgr: np.ndarray,
                            split: str = "train",
                            filename: str = None) -> str:
    """
    Save a captured frame into the dataset directory for fine-tuning.
    split: 'train' or 'val'
    Returns saved file path.
    """
    dataset_dir = os.path.join("dataset", "images", split)
    os.makedirs(dataset_dir, exist_ok=True)

    if filename is None:
        filename = f"frame_{int(time.time())}.jpg"

    path = os.path.join(dataset_dir, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def stream_info(youtube_url: str = ABBEY_ROAD_URL) -> dict:
    """
    Get stream metadata using yt-dlp.
    Returns dict with title, uploader, view_count etc.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", youtube_url],
            capture_output=True, text=True, timeout=20
        )
        import json
        data = json.loads(result.stdout)
        return {
            "title": data.get("title", "Unknown"),
            "uploader": data.get("uploader", "Unknown"),
            "view_count": data.get("view_count", 0),
            "is_live": data.get("is_live", False),
            "thumbnail": data.get("thumbnail", ""),
        }
    except Exception as e:
        return {"error": str(e)}