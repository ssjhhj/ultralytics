import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget
)

from train_page import TrainPage
from dataset_class_stat import StatPage
from class_replace_page import ClassReplacePage
from dataset_split_page import DatasetSplitPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 训练与数据集工具")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== 顶部导航栏 =====
        nav_layout = QHBoxLayout()
        btn_train = QPushButton("训练")
        btn_stat = QPushButton("数据集分析")
        btn_replace = QPushButton("Class 替换")
        btn_split = QPushButton("数据集划分")

        nav_layout.addWidget(btn_train)
        nav_layout.addWidget(btn_stat)
        nav_layout.addWidget(btn_replace)
        nav_layout.addWidget(btn_split)
        nav_layout.addStretch()

        main_layout.addLayout(nav_layout)

        # ===== Stack =====
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # ===== 页面 =====
        self.train_page = TrainPage()
        self.stat_page = StatPage()
        self.replace_page = ClassReplacePage()
        self.split_page = DatasetSplitPage()


        self.stack.addWidget(self.train_page)  # index 0
        self.stack.addWidget(self.stat_page)   # index 1
        self.stack.addWidget(self.replace_page) # index 2
        self.stack.addWidget(self.split_page) # index 3

        # ===== 绑定切换 =====
        btn_train.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_stat.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_replace.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        btn_split.clicked.connect(lambda: self.stack.setCurrentIndex(3))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
