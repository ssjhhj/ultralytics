import random
import shutil
import yaml
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QDoubleSpinBox, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt


class DatasetSplitPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_dir = None
        self.label_dir = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ===== 选择目录 =====
        self.img_label = QLabel("未选择图像文件夹")
        self.lbl_label = QLabel("未选择标签文件夹")

        btn_img = QPushButton("选择图像文件夹")
        btn_lbl = QPushButton("选择标签文件夹 (YOLO txt)")

        btn_img.clicked.connect(self.select_image_dir)
        btn_lbl.clicked.connect(self.select_label_dir)

        layout.addLayout(self.row(btn_img, self.img_label))
        layout.addLayout(self.row(btn_lbl, self.lbl_label))

        # ===== 比例设置 =====
        layout.addWidget(QLabel("数据集划分比例（总和必须为 1.0）"))

        self.train_spin = QDoubleSpinBox()
        self.val_spin = QDoubleSpinBox()
        self.test_spin = QDoubleSpinBox()

        for s in (self.train_spin, self.val_spin, self.test_spin):
            s.setRange(0.0, 1.0)
            s.setSingleStep(0.05)
            s.setDecimals(2)

        self.train_spin.setValue(0.8)
        self.val_spin.setValue(0.1)
        self.test_spin.setValue(0.1)

        layout.addLayout(self.row(QLabel("Train"), self.train_spin))
        layout.addLayout(self.row(QLabel("Val"), self.val_spin))
        layout.addLayout(self.row(QLabel("Test"), self.test_spin))

        # ===== 执行 =====
        self.split_btn = QPushButton("开始划分并生成 data.yaml")
        self.split_btn.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_btn)

        # ===== 日志 =====
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def row(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        h.addStretch()
        return h

    # ================= 目录选择 =================

    def select_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if path:
            self.image_dir = Path(path)
            self.img_label.setText(path)

    def select_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        if path:
            self.label_dir = Path(path)
            self.lbl_label.setText(path)

    # ================= 核心逻辑 =================

    def split_dataset(self):
        if not self.image_dir or not self.label_dir:
            QMessageBox.warning(self, "错误", "请先选择图像和标签文件夹")
            return

        train_r = self.train_spin.value()
        val_r = self.val_spin.value()
        test_r = self.test_spin.value()

        if abs(train_r + val_r + test_r - 1.0) > 1e-6:
            QMessageBox.warning(self, "错误", "Train + Val + Test 必须等于 1.0")
            return

        # ---------- 收集有效 image-label ----------
        images = []
        skipped = 0

        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            images.extend(self.image_dir.glob(f"*{ext}"))

        valid = []
        for img in images:
            if (self.label_dir / f"{img.stem}.txt").exists():
                valid.append(img)
            else:
                skipped += 1

        if not valid:
            QMessageBox.warning(self, "错误", "没有任何有效 image-label 对")
            return

        random.shuffle(valid)

        n = len(valid)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        train_imgs = valid[:n_train]
        val_imgs = valid[n_train:n_train + n_val]
        test_imgs = valid[n_train + n_val:]

        # ---------- 创建输出目录 ----------
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = self.image_dir.parent / f"dataset_{time_str}"

        for p in [
            "images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"
        ]:
            (out_root / p).mkdir(parents=True, exist_ok=True)

        class_ids = set()

        def copy_set(imgs, split):
            for img in imgs:
                lbl = self.label_dir / f"{img.stem}.txt"
                shutil.copy2(img, out_root / "images" / split / img.name)
                shutil.copy2(lbl, out_root / "labels" / split / lbl.name)

                with lbl.open() as f:
                    for line in f:
                        if line.strip():
                            class_ids.add(line.split()[0])

        copy_set(train_imgs, "train")
        copy_set(val_imgs, "val")
        copy_set(test_imgs, "test")

        # ---------- data.yaml ----------
        class_ids = sorted(class_ids)
        names = [str(c) for c in class_ids]

        data_yaml = {
            "path": str(out_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": names
        }

        yaml_path = out_root / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True)

        # ---------- 日志 ----------
        self.log.append(
            f"[OK] 数据集生成完成\n"
            f"输出目录: {out_root}\n"
            f"Train: {len(train_imgs)}  Val: {len(val_imgs)}  Test: {len(test_imgs)}"
        )

        if skipped > 0:
            self.log.append(f"[WARN] 跳过 {skipped} 张无标签图片")

        self.log.append(f"data.yaml: {yaml_path}")


# ===== 独立运行 =====
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = DatasetSplitPage()
    w.setWindowTitle("数据集划分工具")
    w.resize(700, 600)
    w.show()
    sys.exit(app.exec_())
