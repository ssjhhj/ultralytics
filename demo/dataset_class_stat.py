import os
import random
from collections import Counter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

class StatPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.label_path = QLabel("未选择标签文件夹")
        btn_select = QPushButton("选择标签文件夹")
        btn_select.clicked.connect(self.select_label_dir)

        top.addWidget(self.label_path)
        top.addWidget(btn_select)

        self.fig = Figure()
        self.canvas = Canvas(self.fig)

        layout.addLayout(top)
        layout.addWidget(self.canvas)

    def select_label_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择 YOLO 标签文件夹")
        if not dir_path:
            return

        self.label_path.setText(dir_path)
        self.draw_pie(dir_path)

    def draw_pie(self, label_dir):
        from collections import Counter
        counter = Counter()
        bad_lines = 0

        for f in os.listdir(label_dir):
            if not f.endswith(".txt"):
                continue
            if f.lower() == "classes.txt":
                continue

            file_path = os.path.join(label_dir, f)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    first = line.split()[0]

                    # ===== 情况 1：标准 YOLO（数字）=====
                    if first.isdigit():
                        counter[f"class_{first}"] += 1

                    # ===== 情况 2：字符串类别 =====
                    else:
                        counter[first] += 1

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if not counter:
            ax.text(0.5, 0.5, "未检测到有效标注",
                    ha="center", va="center", fontsize=14)
            self.canvas.draw()
            return

        labels = list(counter.keys())
        sizes = list(counter.values())

        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("数据集类别分布")

        self.canvas.draw()



# ===== 允许独立运行 =====
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = StatPage()
    w.setWindowTitle("数据集类别统计")
    w.resize(600, 600)
    w.show()
    sys.exit(app.exec_())
