import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def random_color(seed):
    random.seed(str(seed))
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )


class SegViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Viewer (YOLOv8-seg / JSON)")
        self.resize(1200, 820)
        self.setFocusPolicy(Qt.StrongFocus)

        self.img_dir = None
        self.ann_dir = None
        self.images = []
        self.index = 0

        self.colors = {}
        self.alpha = 0.35

        self.init_ui()

    def init_ui(self):
        # ---------- 顶部 ----------
        self.btn_img = QPushButton("选择图片文件夹")
        self.btn_ann = QPushButton("选择标注文件夹")
        self.btn_img.clicked.connect(self.select_img_dir)
        self.btn_ann.clicked.connect(self.select_ann_dir)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_img)
        top_layout.addWidget(self.btn_ann)

        # ---------- 图片显示 ----------
        self.image_label = QLabel("请选择图片和标注文件夹")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFocusPolicy(Qt.NoFocus)
        self.image_label.mousePressEvent = lambda e: self.clear_focus()

        # ---------- 底部 ----------
        self.status_label = QLabel("第 0 / 0 张")
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("输入编号")
        self.jump_input.setFixedWidth(120)

        self.jump_btn = QPushButton("跳转")
        self.jump_input.returnPressed.connect(self.jump_to_index)
        self.jump_btn.clicked.connect(self.jump_to_index)
        bottom_layout = QHBoxLayout()
        self.convert_btn = QPushButton("JSON ⇄ TXT 转换")
        self.convert_btn.clicked.connect(self.convert_annotation)

        bottom_layout.addWidget(self.convert_btn)

        bottom_layout.addWidget(self.status_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(QLabel("跳转到："))
        bottom_layout.addWidget(self.jump_input)
        bottom_layout.addWidget(self.jump_btn)

        # ---------- 主布局 ----------
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    # ================= 事件处理 =================

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_D, Qt.Key_Right):
            self.next_image()
        elif event.key() in (Qt.Key_A, Qt.Key_Left):
            self.prev_image()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        self.clear_focus()
        if event.angleDelta().y() > 0:
            self.prev_image()
        else:
            self.next_image()

    def clear_focus(self):
        self.jump_input.clearFocus()
        self.setFocus()

    # ================= 目录 =================

    def select_img_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.img_dir = Path(path)
            self.images = sorted(
                p for p in self.img_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
            )
            self.index = 0
            self.show_current()

    def select_ann_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标注文件夹")
        if path:
            self.ann_dir = Path(path)
            self.index = 0
            self.show_current()

    # ================= 翻页 =================

    def next_image(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_current()

    def prev_image(self):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.show_current()

    def jump_to_index(self):
        if not self.images:
            return
        text = self.jump_input.text().strip()
        if not text.isdigit():
            return
        idx = int(text)
        if 1 <= idx <= len(self.images):
            self.index = idx - 1
            self.show_current()
        self.clear_focus()

    # ================= 显示 =================

    def show_current(self):
        if not self.img_dir or not self.ann_dir or not self.images:
            return

        img_path = self.images[self.index]
        img = cv2.imread(str(img_path))
        if img is None:
            return

        h, w = img.shape[:2]
        canvas = img.copy()

        stem = img_path.stem
        txt_path = self.ann_dir / f"{stem}.txt"
        json_path = self.ann_dir / f"{stem}.json"

        if txt_path.exists():
            self.draw_yolo_seg(canvas, txt_path, w, h)
        elif json_path.exists():
            self.draw_json_seg(canvas, json_path)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(canvas.data, w, h, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )

        self.status_label.setText(
            f"第 {self.index + 1} / {len(self.images)} 张"
        )

    # ================= 绘制 =================

    def draw_yolo_seg(self, img, txt_path, w, h):
        overlay = img.copy()
        with txt_path.open() as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                pts = list(map(float, parts[1:]))

                points = np.array(
                    [[int(pts[i] * w), int(pts[i + 1] * h)]
                     for i in range(0, len(pts), 2)],
                    np.int32
                )

                color = self.get_color(cls)
                cv2.fillPoly(overlay, [points], color)
                cv2.polylines(img, [points], True, color, 2)

        cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0, img)

    def draw_json_seg(self, img, json_path):
        overlay = img.copy()
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue
            points = np.array(shape["points"], dtype=np.int32)
            color = self.get_color(shape.get("label"))
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(img, [points], True, color, 2)

        cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0, img)

    # ================= 颜色 =================

    def get_color(self, key):
        if key not in self.colors:
            self.colors[key] = random_color(key)
        return self.colors[key]
    
    # ================= 格式转换 =================
    def convert_annotation(self):
        if not self.ann_dir or not self.img_dir:
            return

        json_files = list(self.ann_dir.glob("*.json"))
        txt_files = list(self.ann_dir.glob("*.txt"))

        # ---------- JSON -> YOLO ----------
        if json_files:
            out_dir = self.ann_dir / "yolo"
            out_dir.mkdir(exist_ok=True)

            for json_path in json_files:
                img_path = self.img_dir / f"{json_path.stem}.jpg"
                if not img_path.exists():
                    img_path = self.img_dir / f"{json_path.stem}.png"
                if not img_path.exists():
                    continue

                out_txt = out_dir / f"{json_path.stem}.txt"
                json_to_yolo_seg(json_path, img_path, out_txt)

            self.ann_dir = out_dir
            self.show_current()
            return

        # ---------- YOLO -> JSON ----------
        if txt_files:
            out_dir = self.ann_dir / "json"
            out_dir.mkdir(exist_ok=True)

            for txt_path in txt_files:
                img_path = self.img_dir / f"{txt_path.stem}.jpg"
                if not img_path.exists():
                    img_path = self.img_dir / f"{txt_path.stem}.png"
                if not img_path.exists():
                    continue

                out_json = out_dir / f"{txt_path.stem}.json"
                yolo_seg_to_json(txt_path, img_path, out_json)

            self.ann_dir = out_dir
            self.show_current()



def json_to_yolo_seg(json_path, img_path, save_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue

        label = shape.get("label", "0")
        try:
            cls = int(label)
        except:
            cls = abs(hash(label)) % 1000

        norm = []
        for x, y in shape["points"]:
            norm.append(f"{x / w:.6f}")
            norm.append(f"{y / h:.6f}")

        lines.append(" ".join([str(cls)] + norm))

    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def yolo_seg_to_json(txt_path, img_path, save_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    shapes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            pts = list(map(float, parts[1:]))

            points = []
            for i in range(0, len(pts), 2):
                points.append([
                    pts[i] * w,
                    pts[i + 1] * h
                ])

            shapes.append({
                "label": str(cls),
                "shape_type": "polygon",
                "points": points
            })

    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SegViewer()
    viewer.show()
    sys.exit(app.exec_())
