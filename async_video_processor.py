import cv2
import threading
import time
from queue import Queue
from detector import detect_frame
from database import insert_detection, insert_event


class AsyncVideoProcessor:
    def __init__(self, max_queue=50):
        self.queue = Queue(maxsize=max_queue)
        self.running = False
        self.thread = None

    def start(self, video_path, max_frames=300, delay=0.01):
        self.running = True
        self.thread = threading.Thread(
            target=self._process_video,
            args=(video_path, max_frames, delay),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False

    def _process_video(self, video_path, max_frames, delay):
        cap = cv2.VideoCapture(video_path)

        frame_id = 0

        while cap.isOpened() and self.running and frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            ann, p, c, o, anomaly, conf = detect_frame(frame)

            # DB save
            insert_detection(p, c, o, anomaly, conf, source="video_async")

            if anomaly:
                insert_event(
                    "VIDEO_ANOMALY",
                    f"Frame {frame_id}: crowd detected ({p} persons)",
                    "warning"
                )

            # push to queue (for UI preview)
            if not self.queue.full():
                self.queue.put({
                    "frame_id": frame_id,
                    "frame": ann,
                    "persons": p,
                    "cars": c,
                    "anomaly": anomaly,
                    "confidence": conf
                })

            frame_id += 1
            time.sleep(delay)

        cap.release()
        self.running = False

    def get_latest(self):
        if self.queue.empty():
            return None
        return self.queue.get()