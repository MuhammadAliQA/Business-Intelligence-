import cv2
import threading
import queue
import time


class VideoStreamEngine:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.queue = queue.Queue(maxsize=1)
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except:
                    pass

            self.queue.put(frame)
            time.sleep(0.03)

    def read(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def stop(self):
        self.running = False
        self.cap.release()