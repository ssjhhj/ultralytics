import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QLineEdit, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt


class ClassReplacePage(QWidget):
    def __init__(self):
        super().__init__()
        self.rule_rows = []
        self.label_dir = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ===== 选择标签文件夹 =====
        top = QHBoxLayout()
        self.lbl_path = QLabel("未选择标签文件夹")
        btn_select = QPushButton("选择标签文件夹")
        btn_select.clicked.connect(self.select_label_dir)

        top.addWidget(self.lbl_path)
        top.addWidget(btn_select)
        layout.addLayout(top)

        # ===== 替换规则区 =====
        layout.addWidget(QLabel("Class 替换规则（old → new）"))

        self.rule_layout = QVBoxLayout()
        layout.addLayout(self.rule_layout)

        # 默认加一行
        self.add_rule_row()

        # ===== 控制按钮 =====
        ctrl = QHBoxLayout()
        btn_add = QPushButton("+ 添加规则")
        btn_clear = QPushButton("清空规则")
        btn_apply = QPushButton("执行替换")

        btn_add.clicked.connect(self.add_rule_row)
        btn_clear.clicked.connect(self.clear_rules)
        btn_apply.clicked.connect(self.apply_replace)

        ctrl.addWidget(btn_add)
        ctrl.addWidget(btn_clear)
        ctrl.addStretch()
        ctrl.addWidget(btn_apply)

        layout.addLayout(ctrl)

        # ===== 日志 =====
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    # ================= UI 行 =================

    def add_rule_row(self):
        row = QHBoxLayout()

        old_edit = QLineEdit()
        old_edit.setPlaceholderText("原 class")

        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignCenter)

        new_edit = QLineEdit()
        new_edit.setPlaceholderText("新 class")

        row.addWidget(old_edit)
        row.addWidget(arrow)
        row.addWidget(new_edit)

        self.rule_layout.addLayout(row)
        self.rule_rows.append((old_edit, new_edit))

    def clear_rules(self):
        for old, new in self.rule_rows:
            old.clear()
            new.clear()

    # ================= 逻辑 =================

    def select_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 标签文件夹")
        if path:
            self.label_dir = path
            self.lbl_path.setText(path)

    def apply_replace(self):
        if not self.label_dir:
            QMessageBox.warning(self, "错误", "请先选择标签文件夹")
            return

        # 构建映射表
        mapping = {}
        for old, new in self.rule_rows:
            o = old.text().strip()
            n = new.text().strip()
            if o and n:
                mapping[o] = n

        if not mapping:
            QMessageBox.warning(self, "错误", "未配置任何替换规则")
            return

        replaced_files = 0
        replaced_lines = 0

        for fname in os.listdir(self.label_dir):
            if not fname.endswith(".txt"):
                continue
            if fname.lower() == "classes.txt":
                continue

            path = os.path.join(self.label_dir, fname)
            changed = False
            new_lines = []

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        new_lines.append(line)
                        continue

                    cls = parts[0]
                    if cls in mapping:
                        parts[0] = mapping[cls]
                        changed = True
                        replaced_lines += 1

                    new_lines.append(" ".join(parts) + "\n")

            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                replaced_files += 1

        self.log.append(
            f"[OK] 完成替换\n"
            f"规则数: {len(mapping)}\n"
            f"影响文件: {replaced_files}\n"
            f"替换标注: {replaced_lines}"
        )


# ===== 允许独立运行 =====
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = ClassReplacePage()
    w.setWindowTitle("Class 替换工具")
    w.resize(700, 500)
    w.show()
    sys.exit(app.exec_())
