import os


def save_yolo_label(path, persons, cars, others):
    label_path = path.replace("images", "labels").replace(".jpg", ".txt")

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    lines = []

    # class 0 = person
    for _ in range(persons):
        lines.append("0 0.5 0.5 0.2 0.4")

    # class 1 = car
    for _ in range(cars):
        lines.append("1 0.5 0.5 0.3 0.3")

    # class 2 = other
    for _ in range(others):
        lines.append("2 0.5 0.5 0.2 0.2")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))