from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.password_field = QLineEdit()
        self.password_field.setEchoMode(QLineEdit.Password)  # Set echo mode to Password
        layout.addWidget(self.password_field)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication([])
    window = MyWidget()
    window.show()
    app.exec_()
