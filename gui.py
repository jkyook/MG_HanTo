import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget,
                             QPushButton, QLineEdit, QComboBox, QToolBar, QAction, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization")
        self.canvas = FigureCanvas(plt.Figure())
        self.selected_file = ''
        self.data = pd.DataFrame()

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Toolbar setup
        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_action)

        # Add save button to toolbar for saving plots
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_plot)
        toolbar.addAction(save_action)

        # Add text boxes for data range input with tooltip
        self.start_index_input = QLineEdit()
        self.start_index_input.setPlaceholderText("Start Index")
        self.start_index_input.setToolTip("Enter the start index of the data range to plot")
        toolbar.addWidget(self.start_index_input)

        self.end_index_input = QLineEdit()
        self.end_index_input.setPlaceholderText("End Index")
        self.end_index_input.setToolTip("Enter the end index of the data range to plot")
        toolbar.addWidget(self.end_index_input)

        self.comboboxes = [QComboBox() for _ in range(8)]
        for combobox in self.comboboxes:
            toolbar.addWidget(combobox)
        draw_button = QPushButton("Draw")
        draw_button.clicked.connect(self.plot_data)
        toolbar.addWidget(draw_button)

    def open_file_dialog(self):
        self.selected_file, _ = QFileDialog.getOpenFileName(self, "Open file", "", "CSV files (*.csv)")
        if self.selected_file:
            self.load_data()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.selected_file)
            sorted_columns = sorted(self.data.columns)  # 데이터 컬럼을 알파벳 순서로 정렬
            for combobox in self.comboboxes:
                combobox.clear()
                combobox.addItems(sorted_columns)  # 정렬된 컬럼 목록을 콤보박스에 추가
            # Set default values for specific comboboxes if columns exist
            default_values = ['price', 'bns_check_3', 'price', 'gray_strong', 'price', 'cover_ordered', 'price', 'profit_opt']
            for combobox, default_value in zip(self.comboboxes, default_values):
                if default_value in self.data.columns:
                    combobox.setCurrentText(default_value)
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))
            return

    def validate_range(self, start, end, length):
        if start is not None and end is not None:
            if start < 0 or end < 0 or start > end or end > length:
                QMessageBox.warning(self, "Range Error", "Invalid data range entered. Please enter a valid range.")
                return False
        return True

    def plot_data(self):
        try:
            start_index = int(self.start_index_input.text()) if self.start_index_input.text().isdigit() else 0
            end_index = int(self.end_index_input.text()) if self.end_index_input.text().isdigit() else len(self.data)
            if not self.validate_range(start_index, end_index, len(self.data)):
                return
            selected_columns = [combobox.currentText() for combobox in self.comboboxes]
            data_range = self.data.iloc[start_index:end_index]
            self.draw_plot(selected_columns, data_range)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while plotting the data: {e}")

    def draw_plot(self, columns, data_range):
        self.canvas.figure.clear()
        axes = self.canvas.figure.subplots(4, 1, sharex=True)
        color_pairs = [('skyblue', 'red'), ('skyblue', 'purple'), ('skyblue', 'brown'), ('skyblue', 'gray')]
        special_colors = ['green', 'orange']
        for i, ax in enumerate(axes):
            col1, col2 = columns[i * 2], columns[i * 2 + 1]
            if col1 in data_range.columns and col2 in data_range.columns:
                ax.plot(data_range[col1], label=col1, color=color_pairs[i][0])
                ax.legend(loc='upper left')
                ax2 = ax.twinx()
                if i == 1:
                    ax2.plot(data_range['gray'], label='gray', color=special_colors[0])
                    ax2.plot(data_range['gray_strong'], label='gray_strong', color=special_colors[1])
                ax2.plot(data_range[col2], label=col2, color=color_pairs[i][1])
                ax2.legend(loc='upper right')
        self.canvas.draw()

    def save_plot(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)", options=options)
        if fileName:
            self.canvas.figure.savefig(fileName)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



