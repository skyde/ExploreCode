#!/usr/bin/env python3

import sys
import os

# PyQt Imports
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPlainTextEdit,
    QListWidget,
    QPushButton,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QAction,
    QFileDialog,
    QLineEdit,
    QGroupBox,
    QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# Syntax / Style Imports
import syntax_highlighter
from gui_style import apply_dark_mode

# Import logic-only module
import api

# ------------------ Worker Signals ------------------ #

class WorkerSignals(QObject):
    """
    Defines custom signals for our background worker.
    """
    log = pyqtSignal(str)           # For logging text
    finished = pyqtSignal()         # Signal completion
    data_updated = pyqtSignal(str)  # Let the UI know new files got written

# ------------------ Generation Worker ------------------ #

class GenerationWorker(QThread):
    """
    A QThread-based worker that calls the logic in api.py
    but emits signals to update the GUI.
    """
    signals = WorkerSignals()

    def __init__(self, code_input, prompt_input, parent=None):
        super().__init__(parent)
        self.code_input = code_input
        self.prompt_input = prompt_input

    def run(self):
        """
        Calls into api.run_generation_process, forwarding logs & events via signals.
        """
        def log_func(msg):
            self.signals.log.emit(msg)

        def data_updated_func(folder):
            self.signals.data_updated.emit(folder)

        def finished_func():
            self.signals.finished.emit()

        log_func("[GUI Worker] Starting generation process...\n")

        api.run_generation_process(
            code_input=self.code_input,
            prompt_input=self.prompt_input,
            log_func=log_func,
            data_updated_func=data_updated_func,
            finished_func=finished_func
        )

# ------------------ Main Window ------------------ #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Rancher 🐎")
        self.resize(1200, 800)

        # Apply a dark mode palette
        apply_dark_mode(self)

        # Create the menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Action: Load from Disk
        load_action = QAction("Load from Disk", self)
        load_action.triggered.connect(self.on_load_from_disk)
        file_menu.addAction(load_action)

        # Add Reset action to File menu
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.on_reset)
        file_menu.addAction(reset_action)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---------- TOP: Prompt (left) and Model Settings (right) side by side ---------- #
        top_layout = QHBoxLayout()

        # Prompt group
        prompt_group = QGroupBox("Prompt")
        prompt_group_layout = QVBoxLayout()

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")

        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Generation")
        self.stop_button = QPushButton("Stop Generation")
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.run_button)
        btn_layout.addWidget(self.stop_button)

        prompt_group_layout.addWidget(self.prompt_edit)
        prompt_group_layout.addLayout(btn_layout)
        prompt_group.setLayout(prompt_group_layout)

        # Model Settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()

        self.initial_model_edit = QLineEdit()
        self.initial_model_edit.setText(api.INITAL_MODEL_NAME)
        model_layout.addRow("Initial Model Name:", self.initial_model_edit)

        self.fix_model_edit = QLineEdit()
        self.fix_model_edit.setText(api.FIX_MODEL_NAME)
        model_layout.addRow("Fix Model Name:", self.fix_model_edit)

        self.max_iterations_edit = QLineEdit()
        self.max_iterations_edit.setText(str(api.MAX_ITERATIONS))
        model_layout.addRow("Max Iterations:", self.max_iterations_edit)

        model_group.setLayout(model_layout)

        # Add prompt on the left, model on the right
        top_layout.addWidget(prompt_group, stretch=3)
        top_layout.addWidget(model_group, stretch=2)

        main_layout.addLayout(top_layout)

        # ---------- Main splitter: History | Output | Code side by side ---------- #
        triple_splitter = QSplitter(Qt.Horizontal)

        # History
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout()
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        history_group.setLayout(history_layout)
        triple_splitter.addWidget(history_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        triple_splitter.addWidget(output_group)

        # Code
        code_group = QGroupBox("C++ Code")
        code_layout = QVBoxLayout()
        self.code_edit = QPlainTextEdit()
        code_layout.addWidget(self.code_edit)
        code_group.setLayout(code_layout)
        triple_splitter.addWidget(code_group)

        # Set initial stretch factors so History is smaller
        # and Output/Code share the remaining space.
        triple_splitter.setStretchFactor(0, 1)  # History
        triple_splitter.setStretchFactor(1, 2)  # Output
        triple_splitter.setStretchFactor(2, 2)  # Code

        main_layout.addWidget(triple_splitter, 1)

        # Worker thread placeholder
        self.worker_thread = None
        self._iteration_history = []

        # Connect signals
        self.run_button.clicked.connect(self.start_generation)
        self.stop_button.clicked.connect(self.stop_generation)
        self.history_list.currentRowChanged.connect(self.on_history_select)

        # Attach syntax highlighters
        self.cpp_highlighter = syntax_highlighter.CppSyntaxHighlighter(self.code_edit.document())
        self.output_highlighter = syntax_highlighter.OutputHighlighter(self.output_text.document())

    # -----------------------------------------------

    def on_load_from_disk(self):
        folder = QFileDialog.getExistingDirectory(self, "Select session folder to load code")
        if folder:
            self.load_data_into_ui(folder)

    def load_data_into_ui(self, folder: str):
        loaded_history = api.load_iteration_history_from_folder(folder)
        loaded_prompt = api.load_prompt_from_folder(folder)

        self.history_list.clear()
        self._iteration_history.clear()
        self.output_text.clear()

        if loaded_prompt:
            self.prompt_edit.setPlainText(loaded_prompt)

        if loaded_history:
            self._iteration_history = loaded_history
            for item in self._iteration_history:
                self.history_list.addItem(item['label'])

            # Show the last iteration by default
            last_index = len(self._iteration_history) - 1
            self.history_list.setCurrentRow(last_index)
            self.on_history_select(last_index)

            self.append_log(f"[GUI] Loaded {len(loaded_history)} iteration(s) + prompt from '{folder}'\n")
        else:
            if loaded_prompt:
                self.append_log(f"[GUI] No iteration files found, but loaded prompt from '{folder}'\n")
            else:
                self.append_log(f"[GUI] No iteration files or prompt found in '{folder}'\n")

    def start_generation(self):
        if self.worker_thread and self.worker_thread.isRunning():
            return

        code_input = self.code_edit.toPlainText()
        prompt_input = self.prompt_edit.toPlainText()

        self.history_list.clear()
        self._iteration_history.clear()
        self.output_text.clear()

        # Update global API config based on UI
        api.INITAL_MODEL_NAME = self.initial_model_edit.text().strip()
        api.FIX_MODEL_NAME = self.fix_model_edit.text().strip()
        try:
            api.MAX_ITERATIONS = int(self.max_iterations_edit.text().strip())
        except ValueError:
            api.MAX_ITERATIONS = 10

        # Create the worker
        self.worker_thread = GenerationWorker(code_input, prompt_input)
        self.worker_thread.signals.log.connect(self.append_log)
        self.worker_thread.signals.data_updated.connect(self.on_data_updated_from_worker)
        self.worker_thread.signals.finished.connect(self.generation_finished)
        self.worker_thread.started.connect(self.on_generation_started)
        self.worker_thread.start()

    def on_generation_started(self):
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_generation(self):
        api.stop_generation()
        self.append_log("[GUI] Stop signal sent to logic.\n")

    def generation_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.append_log("[GUI] Generation thread finished.\n")

    def on_data_updated_from_worker(self, session_folder):
        self.load_data_into_ui(session_folder)

    def append_log(self, message):
        self.output_text.moveCursor(self.output_text.textCursor().End)
        self.output_text.insertPlainText(message)
        self.output_text.moveCursor(self.output_text.textCursor().End)
        QApplication.processEvents()

    def on_history_select(self, index):
        if index < 0 or index >= len(self._iteration_history):
            return

        data = self._iteration_history[index]
        self.code_edit.setPlainText(data['code'])
        self.output_text.clear()
        self.output_text.insertPlainText(data['output'])

    def on_reset(self):
        api.reset_state()
        self.max_iterations_edit.setText(str(api.MAX_ITERATIONS))
        self.initial_model_edit.setText(api.INITAL_MODEL_NAME)
        self.fix_model_edit.setText(api.FIX_MODEL_NAME)
        self.prompt_edit.clear()
        self.code_edit.clear()
        self.output_text.clear()
        self.history_list.clear()

def gui_main_qt():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    gui_main_qt()
