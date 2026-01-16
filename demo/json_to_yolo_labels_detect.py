import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_or_init_classes(classes_file: Path) -> Dict[str, int]:
    class_to_id: Dict[str, int] = {}
    if classes_file.exists():
        with classes_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    class_to_id[name] = idx
    return class_to_id


def save_classes(classes_file: Path, class_to_id: Dict[str, int]) -> None:
    # Ensure order by id
    id_to_class: List[Tuple[int, str]] = sorted(((i, n) for n, i in class_to_id.items()), key=lambda x: x[0])
    classes_file.parent.mkdir(parents=True, exist_ok=True)
    with classes_file.open("w", encoding="utf-8") as f:
        for _, name in id_to_class:
            f.write(f"{name}\n")


def ensure_class_id(class_to_id: Dict[str, int], class_name: str) -> int:
    if class_name in class_to_id:
        return class_to_id[class_name]
    new_id = len(class_to_id)
    class_to_id[class_name] = new_id
    return new_id


def rectangle_to_yolo(points: List[List[float]], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # points: [[x1, y1], [x2, y2]]
    if not points or len(points) < 2:
        raise ValueError("Rectangle points should have at least 2 points: [[x1,y1],[x2,y2]]")
    x1, y1 = points[0]
    x2, y2 = points[1]
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)

    # Clamp to image bounds
    x_min = max(0.0, min(float(img_w), float(x_min)))
    x_max = max(0.0, min(float(img_w), float(x_max)))
    y_min = max(0.0, min(float(img_h), float(y_min)))
    y_max = max(0.0, min(float(img_h), float(y_max)))

    bbox_w = max(0.0, x_max - x_min)
    bbox_h = max(0.0, y_max - y_min)
    cx = x_min + bbox_w / 2.0
    cy = y_min + bbox_h / 2.0

    # Normalize to [0,1]
    x = cx / float(img_w) if img_w else 0.0
    y = cy / float(img_h) if img_h else 0.0
    w = bbox_w / float(img_w) if img_w else 0.0
    h = bbox_h / float(img_h) if img_h else 0.0

    # Final clamp
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return x, y, w, h


def convert_one(json_path: Path, labels_dir: Path, class_to_id: Dict[str, int]) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = int(data.get("size", {}).get("width", 0))
    img_h = int(data.get("size", {}).get("height", 0))
    objects = data.get("objects", []) or []

    yolo_lines: List[str] = []
    for obj in objects:
        geometry_type = obj.get("geometryType")
        class_title = obj.get("classTitle") or str(obj.get("classId", "unknown"))
        if geometry_type != "rectangle":
            # 非矩形标注暂不处理
            continue
        points = (obj.get("points", {}) or {}).get("exterior", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        try:
            x, y, w, h = rectangle_to_yolo(points, img_w, img_h)
        except Exception:
            continue
        cls_id = ensure_class_id(class_to_id, class_title)
        yolo_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if yolo_lines:
        # labels 文件名与图片同名：如果 json 是 name.jpg.json，则生成 name.txt
        # Path.stem 去掉最后一个扩展名，这里需两次以去掉 .json 和 .jpg/.png
        first_stem = json_path.stem  # e.g., bisturi2.jpg
        pure_stem = Path(first_stem).stem  # e.g., bisturi2
        out_txt = labels_dir / f"{pure_stem}.txt"
        labels_dir.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert DatasetNinja-style JSON rectangles to YOLO txt labels")
    parser.add_argument("--ann-dir", type=str, required=False, help="JSON 标注目录，例如 F:/Downloads/.../train/ann",default=r"F:\Downloads\bee-image-object-detection-DatasetNinja\ds\ann")
    parser.add_argument("--img-dir", type=str, required=False, help="图片目录（可选，仅用于对齐命名）",default=r"F:\Downloads\bee-image-object-detection-DatasetNinja\ds\img")
    parser.add_argument("--labels-dir", type=str, required=False, help="输出 YOLO 标签目录，例如 F:/Downloads/.../train/labels",default=r"F:\Downloads\bee-image-object-detection-DatasetNinja\ds\labels")
    parser.add_argument("--classes", type=str, default=None, help="classes.txt 路径（默认写到 labels 目录）")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    labels_dir = Path(args.labels_dir)
    classes_file = Path(args.classes) if args.classes else (labels_dir / "classes.txt")

    if not ann_dir.exists() or not ann_dir.is_dir():
        raise FileNotFoundError(f"ann-dir 不存在或不是目录: {ann_dir}")

    class_to_id = load_or_init_classes(classes_file)

    json_files = sorted(ann_dir.glob("*.json"))
    total = len(json_files)
    converted = 0

    for jp in json_files:
        convert_one(jp, labels_dir, class_to_id)
        converted += 1

    # 写回/更新 classes.txt
    save_classes(classes_file, class_to_id)

    print(f"JSON 总数: {total}, 已处理: {converted}. 输出目录: {labels_dir}")
    print(f"classes.txt: {classes_file}")


if __name__ == "__main__":
    main()


