import sys
import random
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ultralytics import YOLO
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "demo")) 

from dataset_class_stat import DatasetStatGUI


# ================= 数据集划分 + data.yaml =================

def split_and_generate_dataset(image_dir: Path, label_dir: Path, train_ratio=0.8):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    images = []
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        images.extend(image_dir.glob(f"*{ext}"))

    valid_pairs = []
    skipped = 0

    for img in images:
        if (label_dir / f"{img.stem}.txt").exists():
            valid_pairs.append(img)
        else:
            skipped += 1

    if not valid_pairs:
        raise RuntimeError("没有找到任何有效的 image-label 对")

    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * train_ratio)
    train_imgs = valid_pairs[:split_idx]
    test_imgs = valid_pairs[split_idx:]

    date_str = datetime.now().strftime("%Y%m%d")
    out_root = image_dir.parent / f"dataset_{date_str}"

    for p in [
        "images/train", "images/test",
        "labels/train", "labels/test"
    ]:
        (out_root / p).mkdir(parents=True, exist_ok=True)

    class_ids = set()

    def copy_set(imgs, split):
        for img in imgs:
            label = label_dir / f"{img.stem}.txt"
            shutil.copy2(img, out_root / "images" / split / img.name)
            shutil.copy2(label, out_root / "labels" / split / label.name)

            with label.open() as f:
                for line in f:
                    class_ids.add(int(line.split()[0]))

    copy_set(train_imgs, "train")
    copy_set(test_imgs, "test")

    # ---------- data.yaml ----------
    class_ids = sorted(class_ids)
    nc = len(class_ids)
    names = [f"class{i}" for i in range(nc)]

    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "nc": nc,
        "names": names
    }

    yaml_path = out_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    return {
        "out_dir": out_root,
        "yaml": yaml_path,
        "train": len(train_imgs),
        "test": len(test_imgs),
        "skipped": skipped
    }


# ================= 训练线程 =================

class TrainThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, model_path, data_yaml, epochs, batch, imgsz, device):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device

    def run(self):
        log_file = Path.cwd() / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        def logger(msg):
            self.log_signal.emit(msg)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        try:
            logger("=== Training Started ===")
            model = YOLO(self.model_path, task="segment")

            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch,
                imgsz=self.imgsz,
                device=self.device,
                verbose=True
            )

            logger("=== Training Finished ===")
            logger(f"Log saved to: {log_file}")

        except Exception as e:
            logger(f"[ERROR] {e}")


# ================= GUI =================

class TrainerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8-Seg Trainer")
        self.resize(820, 700)

        self.image_dir = ""
        self.label_dir = ""

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # -------- 数据集划分 --------
        self.img_btn = QPushButton("选择图像文件夹")
        self.lbl_btn = QPushButton("选择标签文件夹(YOLO txt)")
        self.split_btn = QPushButton("划分数据集 (8:2 + 自动生成 data.yaml)")

        self.img_label = QLabel("未选择")
        self.lbl_label = QLabel("未选择")

        self.img_btn.clicked.connect(self.select_img)
        self.lbl_btn.clicked.connect(self.select_lbl)
        self.split_btn.clicked.connect(self.split_dataset)

        layout.addLayout(self.row(self.img_btn, self.img_label))
        layout.addLayout(self.row(self.lbl_btn, self.lbl_label))
        layout.addWidget(self.split_btn)

        # -------- 训练参数 --------
        self.model_edit = QLineEdit("yolo11n-seg.pt")
        self.model_btn = QPushButton("选择模型")
        self.data_edit = QLineEdit()
        self.data_btn = QPushButton("选择 YAML")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setValue(200)
        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(32)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setValue(640)
        self.device_edit = QLineEdit("0")
        self.stat_btn = QPushButton("数据集类别分析")
        self.stat_btn.clicked.connect(self.open_stat_window)
        layout.addWidget(self.stat_btn)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_btn)

        layout.addWidget(QLabel("模型路径"))
        layout.addLayout(model_layout)
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_edit)
        data_layout.addWidget(self.data_btn)
        layout.addWidget(QLabel("data.yaml 路径"))
        layout.addLayout(data_layout)

        layout.addLayout(self.row(QLabel("Epochs"), self.epochs_spin))
        layout.addLayout(self.row(QLabel("Batch"), self.batch_spin))
        layout.addLayout(self.row(QLabel("ImgSz"), self.imgsz_spin))
        layout.addLayout(self.row(QLabel("Device"), self.device_edit))

        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_train)
        layout.addWidget(self.train_btn)

        # -------- 日志 --------
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.setLayout(layout)
        self.model_btn.clicked.connect(self.select_model_path)
        self.data_btn.clicked.connect(self.select_data_yaml)

    def row(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    def open_stat_window(self):
        self.stat_win = DatasetStatGUI()
        self.stat_win.show()

    def select_model_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "YOLO Model (*.pt)"
        )
        if path:
            self.model_edit.setText(path)

    def select_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 data.yaml",
            "",
            "YAML (*.yaml *.yml)"
        )
        if path:
            self.data_edit.setText(path)

    def select_img(self):
        p = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if p:
            self.image_dir = p
            self.img_label.setText(Path(p).name)

    def select_lbl(self):
        p = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        if p:
            self.label_dir = p
            self.lbl_label.setText(Path(p).name)

    def split_dataset(self):
        if not self.image_dir or not self.label_dir:
            self.log_view.append("[ERROR] 请先选择图像和标签文件夹")
            return

        info = split_and_generate_dataset(
            Path(self.image_dir),
            Path(self.label_dir)
        )

        self.data_edit.setText(str(info["yaml"]))

        self.log_view.append(
            f"[INFO] 数据集生成完成\n"
            f"Train: {info['train']}  Test: {info['test']}"
        )

        if info["skipped"] > 0:
            self.log_view.append(
                f"[WARN] 有 {info['skipped']} 张图片没有标签，已自动跳过"
            )

    def start_train(self):
        self.thread = TrainThread(
            self.model_edit.text(),
            self.data_edit.text(),
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.imgsz_spin.value(),
            self.device_edit.text()
        )
        self.thread.log_signal.connect(self.log_view.append)
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TrainerGUI()
    gui.show()
    sys.exit(app.exec_())
