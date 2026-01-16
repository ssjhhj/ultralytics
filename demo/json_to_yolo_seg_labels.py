import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def load_or_init_classes(classes_file: Path) -> Dict[str, int]:
    class_to_id = {}
    if classes_file.exists():
        with classes_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    class_to_id[name] = idx
    return class_to_id


def save_classes(classes_file: Path, class_to_id: Dict[str, int]) -> None:
    classes_file.parent.mkdir(parents=True, exist_ok=True)
    id_to_class = sorted(class_to_id.items(), key=lambda x: x[1])
    with classes_file.open("w", encoding="utf-8") as f:
        for name, _id in id_to_class:
            f.write(f"{name}\n")


def ensure_class_id(class_to_id: Dict[str, int], name: str) -> int:
    if name not in class_to_id:
        class_to_id[name] = len(class_to_id)
    return class_to_id[name]


def normalize_polygon(points: List[List[float]], w: int, h: int) -> List[float]:
    norm = []
    for x, y in points:
        nx = min(max(x / w, 0.0), 1.0)
        ny = min(max(y / h, 0.0), 1.0)
        norm.extend([nx, ny])
    return norm


def convert_one(
    json_path: Path,
    img_dir: Path,
    labels_dir: Path,
    class_to_id: Dict[str, int],
) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        return

    # 推断图片路径（json 与图片同名）
    img_path = None
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        p = img_dir / (json_path.stem + ext)
        if p.exists():
            img_path = p
            break

    if img_path is None:
        print(f"[WARN] 未找到对应图片: {json_path.name}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 读取图片失败: {img_path}")
        return

    h, w = img.shape[:2]

    yolo_lines = []

    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue

        label = shape.get("label")
        points = shape.get("points", [])
        if not label or len(points) < 3:
            continue

        cls_id = ensure_class_id(class_to_id, label)
        norm_points = normalize_polygon(points, w, h)

        line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in norm_points)
        yolo_lines.append(line)

    if yolo_lines:
        labels_dir.mkdir(parents=True, exist_ok=True)
        out_txt = labels_dir / f"{json_path.stem}.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")


def main():
    parser = argparse.ArgumentParser("Labelme JSON → YOLOv8-seg")
    parser.add_argument("--ann-dir", required=False, help="Labelme JSON 目录",default="D:/Desktop/XLWD/dataset/2025_12_29_split/Received/1")
    parser.add_argument("--img-dir", required=False, help="图片目录",default="D:/Desktop/XLWD/dataset/2025_12_29_split/Received/1")
    parser.add_argument("--labels-dir", required=False, help="输出 labels 目录",default="D:/Desktop/XLWD/dataset/2025_12_29_split/Received/1_yolo")
    parser.add_argument("--classes", default=None, help="classes.txt 路径")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    labels_dir = Path(args.labels_dir)
    classes_file = Path(args.classes) if args.classes else labels_dir / "classes.txt"

    class_to_id = load_or_init_classes(classes_file)

    json_files = sorted(ann_dir.glob("*.json"))
    for jp in json_files:
        convert_one(jp, img_dir, labels_dir, class_to_id)

    save_classes(classes_file, class_to_id)

    print(f"转换完成，共处理 {len(json_files)} 个 JSON")
    print(f"labels 输出目录: {labels_dir}")
    print(f"classes.txt: {classes_file}")


if __name__ == "__main__":
    main()
