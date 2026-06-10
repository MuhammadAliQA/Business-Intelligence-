import os, time, shutil, zipfile
from pathlib import Path
from database import insert_fine_tune_log, insert_event

DATASET_DIR = "custom_dataset"
RUNS_DIR = "runs"

def prepare_dataset_structure():
    """Create YOLO dataset folder structure."""
    for split in ["train", "val"]:
        os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

def save_uploaded_dataset(zip_file_bytes):
    """
    Accept a zip file containing images+labels, extract to dataset dir.
    Expected zip structure:
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
    """
    prepare_dataset_structure()
    zip_path = f"{DATASET_DIR}/upload.zip"
    with open(zip_path, "wb") as f:
        f.write(zip_file_bytes)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATASET_DIR)
    os.remove(zip_path)
    return True

def create_data_yaml():
    """Write dataset YAML config for YOLO training."""
    yaml_content = f"""path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val
nc: 1
names: ['person']
"""
    yaml_path = f"{DATASET_DIR}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def count_dataset_images():
    counts = {}
    for split in ["train", "val"]:
        img_dir = Path(f"{DATASET_DIR}/images/{split}")
        if img_dir.exists():
            counts[split] = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        else:
            counts[split] = 0
    return counts

def run_fine_tuning(epochs=10, batch=8, imgsz=640, base_model="yolov8n.pt", progress_callback=None):
    """
    Run YOLOv8 fine-tuning. Logs each epoch to DB.
    progress_callback(epoch, total, loss, acc) for Streamlit progress bar.
    Returns: best model path or None
    """
    from ultralytics import YOLO
    import random

    yaml_path = create_data_yaml()
    counts = count_dataset_images()

    if counts.get("train", 0) == 0:
        insert_event("FINE_TUNE", "No training images found — using simulated training", "warning")
        # Simulated training for demo
        for ep in range(1, epochs + 1):
            time.sleep(0.3)
            loss = round(1.0 - (ep / epochs) * 0.7 + random.uniform(-0.05, 0.05), 4)
            acc = round(0.5 + (ep / epochs) * 0.4 + random.uniform(-0.03, 0.03), 4)
            insert_fine_tune_log(ep, loss, acc, "simulated")
            if progress_callback:
                progress_callback(ep, epochs, loss, acc)
        insert_event("FINE_TUNE", f"Simulated fine-tuning completed ({epochs} epochs)", "info")
        return None

    try:
        model = YOLO(base_model)
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=RUNS_DIR,
            name="assbi_finetune",
            exist_ok=True,
            verbose=False
        )

        # Log each epoch result
        for ep, (loss, acc) in enumerate(zip(
            results.results_dict.get("train/box_loss", []),
            results.results_dict.get("metrics/precision(B)", [])
        ), 1):
            insert_fine_tune_log(ep, round(loss, 4), round(acc, 4),
                                 str(results.save_dir / "weights/best.pt"))
            if progress_callback:
                progress_callback(ep, epochs, round(loss, 4), round(acc, 4))

        best_path = str(results.save_dir / "weights/best.pt")
        insert_event("FINE_TUNE", f"Fine-tuning done. Best model: {best_path}", "info")
        return best_path

    except Exception as e:
        insert_event("FINE_TUNE_ERROR", str(e), "error")
        return None

def get_best_model_path():
    best = Path(f"{RUNS_DIR}/assbi_finetune/weights/best.pt")
    return str(best) if best.exists() else None