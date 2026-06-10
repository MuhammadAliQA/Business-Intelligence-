import os
from ultralytics import YOLO

DATA = "dataset/data.yaml"
OUT = "fine_tuned_models"

os.makedirs(OUT, exist_ok=True)


def run_fine_tuning(epochs, batch, imgsz, base_model):
    if not os.path.exists(DATA):
        return None

    model = YOLO(base_model)

    model.train(
        data=DATA,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=OUT,
        name="run",
        exist_ok=True
    )

    best = f"{OUT}/run/weights/best.pt"
    return best if os.path.exists(best) else None


def count_dataset_images():
    def count(p):
        return len(os.listdir(p)) if os.path.exists(p) else 0

    return {
        "train": count("dataset/images/train"),
        "val": count("dataset/images/val")
    }


def get_best_model_path():
    path = "fine_tuned_models/run/weights/best.pt"
    return path if os.path.exists(path) else None


def save_uploaded_dataset(zip_bytes):
    import zipfile

    path = "/tmp/ds.zip"
    open(path, "wb").write(zip_bytes)

    with zipfile.ZipFile(path, "r") as z:
        z.extractall("dataset")