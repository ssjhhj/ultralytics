from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import datetime
import sys
import os


class TrainWorker(QThread):
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
        log_file = f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            def log(msg):
                self.log_signal.emit(msg)
                f.write(msg + "\n")

            try:
                log("=== Training Started ===")
                model = YOLO(self.model_path, task="segment")
                model.train(
                    data=self.data_yaml,
                    epochs=int(self.epochs),
                    batch=int(self.batch),
                    imgsz=int(self.imgsz),
                    device=self.device
                )
                log("=== Training Finished ===")
            except Exception as e:
                log(f"[ERROR] {e}")


class TrainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ===== 模型路径 =====
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        btn_model = QPushButton("选择模型")
        btn_model.clicked.connect(self.select_model)

        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(btn_model)

        # ===== data.yaml =====
        data_layout = QHBoxLayout()
        self.data_edit = QLineEdit()
        btn_data = QPushButton("选择 data.yaml")
        btn_data.clicked.connect(self.select_data)

        data_layout.addWidget(QLabel("data.yaml:"))
        data_layout.addWidget(self.data_edit)
        data_layout.addWidget(btn_data)

        # ===== 参数 =====
        param_layout = QHBoxLayout()
        self.epoch_edit = QLineEdit("200")
        self.batch_edit = QLineEdit("32")
        self.imgsz_edit = QLineEdit("640")

        param_layout.addWidget(QLabel("Epochs"))
        param_layout.addWidget(self.epoch_edit)
        param_layout.addWidget(QLabel("Batch"))
        param_layout.addWidget(self.batch_edit)
        param_layout.addWidget(QLabel("ImgSz"))
        param_layout.addWidget(self.imgsz_edit)

        # ===== 启动 =====
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_train)

        # ===== 日志 =====
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addLayout(model_layout)
        layout.addLayout(data_layout)
        layout.addLayout(param_layout)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.log_box)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "*.pt")
        if path:
            self.model_edit.setText(path)

    def select_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 data.yaml", "", "*.yaml")
        if path:
            self.data_edit.setText(path)

    def start_train(self):
        self.worker = TrainWorker(
            self.model_edit.text(),
            self.data_edit.text(),
            self.epoch_edit.text(),
            self.batch_edit.text(),
            self.imgsz_edit.text(),
            device=0
        )
        self.worker.log_signal.connect(self.log_box.append)
        self.worker.start()
