import cv2
import os

SAVE_DIR = "dataset/frames"
os.makedirs(SAVE_DIR, exist_ok=True)

def capture_burst(url, n_frames=5):

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("❌ Video not opened")
        return []

    frames = []
    saved = 0
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % 15 == 0:   # skip frames
            path = f"{SAVE_DIR}/frame_{saved}.jpg"

            success = cv2.imwrite(path, frame)

            if success:
                print("Saved:", path)
                frames.append(frame)
                saved += 1

            if saved >= n_frames:
                break

        count += 1

    cap.release()

    return frames