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
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# Syntax / Style Imports
import syntax_highlighter
from gui_style import apply_dark_mode

# Import logic-only module
import explore_code

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
    A QThread-based worker that calls the logic in explore_code.py
    but emits signals to update the GUI.
    """
    signals = WorkerSignals()

    def __init__(self, code_input, prompt_input, parent=None):
        super().__init__(parent)
        self.code_input = code_input
        self.prompt_input = prompt_input
        self.session_folder = None

    def run(self):
        """
        Calls into explore_code.run_generation_process(...) so we
        don't keep logic in the GUI. We only forward logs & events via signals.
        """
        # 1) Create a new session folder
        timestamp = explore_code.datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snippet = explore_code.helpers.sanitize_for_filename(self.prompt_input)
        if snippet.strip():
            self.session_folder = os.path.join(
                explore_code.GENERATED_ROOT_FOLDER,
                f"session_{timestamp}_{snippet}"
            )
        else:
            self.session_folder = os.path.join(
                explore_code.GENERATED_ROOT_FOLDER,
                f"session_{timestamp}"
            )
        os.makedirs(self.session_folder, exist_ok=True)

        # 2) "Everything" file accumulates logs
        EVERYTHING_FILE = os.path.join(self.session_folder, "everything.cpp")

        # 3) Define local callbacks so we can pass them to run_generation_process
        def log_func(msg):
            self.signals.log.emit(msg)

        def data_updated_func(folder):
            self.signals.data_updated.emit(folder)

        def finished_func():
            self.signals.finished.emit()

        # 4) Actually run the generation logic
        log_func("[GUI Worker] Starting generation process...\n")
        explore_code.run_generation_process(
            code_input=self.code_input,
            prompt_input=self.prompt_input,
            session_folder=self.session_folder,
            everything_file=EVERYTHING_FILE,
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

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Prompt area
        self.prompt_label = QLabel("Prompt:")
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")

        # Buttons
        self.run_button = QPushButton("Run Generation")
        self.stop_button = QPushButton("Stop Generation")
        self.stop_button.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.run_button)
        btn_layout.addWidget(self.stop_button)

        top_layout = QVBoxLayout()
        top_layout.addWidget(self.prompt_label)
        top_layout.addWidget(self.prompt_edit)
        top_layout.addLayout(btn_layout)

        top_widget = QWidget()
        top_widget.setLayout(top_layout)

        main_layout.addWidget(top_widget)

        # History vs. Output/Code
        splitter_main = QSplitter(Qt.Horizontal)

        # Left (History)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.history_label = QLabel("History:")
        self.history_list = QListWidget()
        left_layout.addWidget(self.history_label)
        left_layout.addWidget(self.history_list)
        splitter_main.addWidget(left_widget)

        # Right -> Another splitter
        splitter_right = QSplitter(Qt.Vertical)

        # Output
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        self.output_label = QLabel("Output:")
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_text)
        output_widget.setLayout(output_layout)
        splitter_right.addWidget(output_widget)

        # Code
        code_widget = QWidget()
        code_layout = QVBoxLayout(code_widget)
        self.code_label = QLabel("C++ Code:")
        self.code_edit = QPlainTextEdit()
        code_layout.addWidget(self.code_label)
        code_layout.addWidget(self.code_edit)
        code_widget.setLayout(code_layout)
        splitter_right.addWidget(code_widget)
        splitter_main.addWidget(splitter_right)
        splitter_main.setStretchFactor(1, 3)

        main_layout.addWidget(splitter_main, 1)

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
        """
        Let the user pick a previously generated folder.
        We read iteration files and prompt from that folder.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select session folder to load code")
        if folder:
            self.load_data_into_ui(folder)

    def load_data_into_ui(self, folder: str):
        """
        Grabs iteration history and prompt from disk, updates UI.
        """
        loaded_history = explore_code.load_iteration_history_from_folder(folder)
        loaded_prompt = explore_code.load_prompt_from_folder(folder)

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
        """
        Spawn a background thread to run the generation logic.
        """
        if self.worker_thread and self.worker_thread.isRunning():
            # Already running, ignore
            return

        code_input = self.code_edit.toPlainText()
        prompt_input = self.prompt_edit.toPlainText()

        # Clear old results
        self.history_list.clear()
        self._iteration_history.clear()
        self.output_text.clear()

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
        """
        Tells the logic side to halt the generation loop.
        """
        explore_code.stop_generation()
        self.append_log("[GUI] Stop signal sent to logic.\n")

    def generation_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.append_log("[GUI] Generation thread finished.\n")

    def on_data_updated_from_worker(self, session_folder):
        """
        Whenever new files are written, reload from disk to keep UI in sync.
        """
        self.load_data_into_ui(session_folder)

    def append_log(self, message):
        """
        Appends a message to the output text box.
        """
        self.output_text.moveCursor(self.output_text.textCursor().End)
        self.output_text.insertPlainText(message)
        self.output_text.moveCursor(self.output_text.textCursor().End)
        QApplication.processEvents()

    def on_history_select(self, index):
        """
        When an iteration is clicked, show its code + output logs.
        """
        if index < 0 or index >= len(self._iteration_history):
            return

        data = self._iteration_history[index]
        self.code_edit.setPlainText(data['code'])
        self.output_text.clear()
        self.output_text.insertPlainText(data['output'])


def gui_main_qt():
    """
    Launches the PyQt application.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui_main_qt()
