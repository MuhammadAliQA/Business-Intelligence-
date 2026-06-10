from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

ANOMALY_THRESHOLD = 8

def detect_frame(frame):
    results = model(frame)[0]

    p=c=o=0
    confs=[]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        confs.append(conf)

        if label == "person":
            p += 1
        elif label in ["car","truck","bus"]:
            c += 1
        else:
            o += 1

    anomaly = p >= ANOMALY_THRESHOLD
    conf = sum(confs)/len(confs) if confs else 0

    return frame, p,c,o, anomaly, conf