#!/usr/bin/env python3

# ------------------ Configuration ------------------ #

USE_DEBUG_PROMPT = False
ALLOW_API_CALLS = False
INITAL_MODEL_NAME = "o1-mini"
FIX_MODEL_NAME = "o1-mini"
MAX_ITERATIONS = 10

GENERATED_ROOT_FOLDER = "generated"
EXECUTABLE = "program"  # We'll compile the code to a program each iteration
PRINT_SEND = True

MAX_PROMPT_LENGTH = 50000

# ------------------ Includes ------------------ #

import os
import sys
import datetime
import re

from ai_service import AIService
import helpers
from helpers import (
    write_to_file, 
    append_to_file, 
    read_file, 
    ensure_prompt_length_ok,
    truncate_preserving_start_and_end
)

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

import syntax_highlighter
from gui_style import apply_dark_mode

# Import the helper class from cpp_compiler.py
from cpp_compiler import CppCompiler

# ------------------ Setup AI Service ------------------ #

ai_service = AIService(not ALLOW_API_CALLS)

# ------------------ Utility: Prompt Length Checker ------------------ #

def ensure_prompt_length_ok(prompt, max_length=MAX_PROMPT_LENGTH):
    """Raise an error if the prompt is too large."""
    if len(prompt) > max_length:
        raise ValueError(f"Prompt length {len(prompt)} exceeded maximum of {max_length} characters. Exiting...")

def maybe_truncate_for_llm(content, max_length=10000):
    """If content is longer than max_length, truncate the middle and add a note."""
    if len(content) <= max_length:
        return content
    half = max_length // 2
    start_part = content[:half]
    end_part = content[-half:]
    return start_part + "\n... [TRUNCATED MIDDLE FOR LLM] ...\n" + end_part

# ------------------ Instantiate the global compiler ------------------ #
cpp_compiler = CppCompiler()

# ------------------ Fix / Compile / Run Helpers ------------------ #

generation_stopped = False

def fix_code_until_success_or_limit(
    initial_code: str,
    session_folder: str,
    everything_file: str,
    label_prefix: str,
    user_problem: str = "",
    fix_prompt: str = helpers.FIX_PROMPT,
    max_iterations: int = MAX_ITERATIONS
):
    """
    Attempt to compile and run initial_code. If it fails either compilation or runtime tests,
    request a fix from OpenAI. Repeat until success or the maximum number of iterations is reached.
    """
    global generation_stopped
    current_code = initial_code

    for fix_iteration in range(1, max_iterations + 1):
        if generation_stopped:
            break

        fix_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}.cpp")
        write_to_file(fix_file, current_code)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} =====\n{current_code}")

        print(f"[main] Compiling {label_prefix} code iteration {fix_iteration}...\n")
        success, error_message = cpp_compiler.compile_cpp(fix_file, os.path.join(session_folder, EXECUTABLE))

        compile_output_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}_compile.txt")
        write_to_file(compile_output_file, error_message)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Output =====\n{error_message}")

        if not success:
            print("[main] Compilation failed. Requesting new fix...\n")
            truncated_compile_log = truncate_preserving_start_and_end(error_message, max_length=7000)
            fix_input = (
                f"{fix_prompt}\n"
                f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                f"Current Code:\n{current_code}"
            )
            ensure_prompt_length_ok(fix_input)
            current_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)
            continue

        print(f"[main] Running {label_prefix} code iteration {fix_iteration}...\n")
        test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

        output_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}_output.txt")
        write_to_file(output_file, test_output)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} Run Output =====\n{test_output}")

        if test_success:
            print(f"[main] {label_prefix} code now passes tests!\n")
            return True, current_code
        else:
            print("[main] Code failed again. Requesting further fix...\n")
            truncated_test_output = truncate_preserving_start_and_end(test_output, max_length=7000)
            fix_input = (
                f"{fix_prompt}\n"
                f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                f"Current Code:\n{current_code}"
            )
            ensure_prompt_length_ok(fix_input)
            current_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)

    return False, current_code

def compile_run_check_code(
    code: str,
    code_filename: str,
    session_folder: str,
    everything_file: str,
    label_prefix: str
):
    """
    Compiles and runs the provided code, returning (success, error_message, runtime_output).
    """
    write_to_file(code_filename, code)
    append_to_file(everything_file, f"===== {label_prefix} Code =====\n{code}")

    print(f"[main] Compiling {label_prefix}...\n")
    success, error_message = cpp_compiler.compile_cpp(code_filename, os.path.join(session_folder, EXECUTABLE))

    compile_output_file = os.path.join(session_folder, f"{label_prefix}_compile.txt")
    write_to_file(compile_output_file, error_message)
    append_to_file(everything_file, f"===== {label_prefix} Compile Output =====\n{error_message}")

    if not success:
        print(f"[main] Error compiling {label_prefix}:\n{error_message}\n")
        return (False, error_message, "")

    print(f"[main] Running {label_prefix}...\n")
    test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

    output_file = os.path.join(session_folder, f"{label_prefix}_output.txt")
    write_to_file(output_file, test_output)
    append_to_file(everything_file, f"===== {label_prefix} Run Output =====\n{test_output}")

    if not test_success:
        print("[main] Code run failed (non-zero exit).")
        return (False, "", test_output)
    else:
        print("[main] Code run succeeded (exit code = 0). Tests passed!\n")
        return (True, "", test_output)

# ------------------ Helper to load last generation from disk ------------------ #

def load_last_generated_code(session_folder: str) -> str:
    """
    Scans the session folder for the highest versioned code file:
    either 'generated_vN.cpp', 'initial_code_fix_vN.cpp', or similar.
    Returns the content of the file with the highest N found.
    If no code is found, returns an empty string.
    """
    if not os.path.isdir(session_folder):
        print(f"[load_last_generated_code] Folder '{session_folder}' does not exist.")
        return ""

    highest_version = -1
    best_file = None

    # Regex for capturing something like _v(\d+).cpp
    version_pattern = re.compile(r"_v(\d+)\.cpp$")

    for filename in os.listdir(session_folder):
        if not filename.endswith(".cpp"):
            continue
        match = version_pattern.search(filename)
        if match:
            version_num = int(match.group(1))
            if version_num > highest_version:
                highest_version = version_num
                best_file = filename

    if best_file:
        full_path = os.path.join(session_folder, best_file)
        print(f"[load_last_generated_code] Loading code from: {full_path}")
        return read_file(full_path)

    print("[load_last_generated_code] No previous generation code found.")
    return ""

# ------------------ PyQt Dark Mode & GUI Implementation ------------------ #

class WorkerSignals(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal()

class GenerationWorker(QThread):
    """
    Runs the generation/fix loop in the background, so the GUI doesn't freeze.
    """
    signals = WorkerSignals()

    def __init__(self, code_input, prompt_input):
        super().__init__()
        self.code_input = code_input
        self.prompt_input = prompt_input
        self.iteration_history = []
        self.session_folder = None

    def run(self):
        global generation_stopped
        generation_stopped = False

        self.signals.log.emit("[GUI] Starting generation process...\n")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build session folder name with a snippet from the prompt so we can see it.
        if self.prompt_input.strip():
            snippet = helpers.sanitize_for_filename(self.prompt_input)
            self.session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_{timestamp}_{snippet}")
        else:
            self.session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_{timestamp}")
        
        os.makedirs(self.session_folder, exist_ok=True)

        EVERYTHING_FILE = os.path.join(self.session_folder, "everything.cpp")
        PROMPT_FILE = os.path.join(self.session_folder, "prompt.txt")

        initial_code_success = False
        single_file_code = ""

        if self.code_input.strip() and not generation_stopped:
            iteration_label = "Initial Code"
            init_code_filename = os.path.join(self.session_folder, "initial_code.cpp")

            success, compile_err, runtime_out = compile_run_check_code(
                self.code_input, init_code_filename, self.session_folder, EVERYTHING_FILE, iteration_label
            )

            iteration_data = {
                'label': iteration_label,
                'code': self.code_input,
                'output': runtime_out if not compile_err else compile_err,
                'compile_err': compile_err,
            }
            self.iteration_history.append(iteration_data)

            if compile_err:
                self.signals.log.emit(f"[GUI] {iteration_label} Compile Error:\n{compile_err}\n\n")
            else:
                self.signals.log.emit(f"[GUI] {iteration_label} Output:\n{runtime_out}\n\n")

            if success:
                initial_code_success = True
            else:
                if compile_err:
                    self.signals.log.emit("[GUI] Fixing compile error...\n")
                    truncated_compile_log = truncate_preserving_start_and_end(compile_err, max_length=7000)
                    fix_input = (
                        f"{helpers.FIX_PROMPT}\n"
                        f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                        f"Current Code:\n{self.code_input}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)

                    if not generation_stopped:
                        success_fix, fixed_code = fix_code_until_success_or_limit(
                            single_file_code, self.session_folder, EVERYTHING_FILE, "initial_code", ""
                        )
                        iteration_data = {
                            'label': "Fixed Initial Code",
                            'code': fixed_code,
                            'output': "Fix pass completed",
                            'compile_err': "",
                        }
                        self.iteration_history.append(iteration_data)
                        self.signals.log.emit("[GUI] Compile fix pass done.\n")

                        if success_fix:
                            initial_code_success = True
                            single_file_code = fixed_code
                else:
                    self.signals.log.emit("[GUI] Fixing runtime error...\n")
                    truncated_test_output = truncate_preserving_start_and_end(runtime_out, max_length=7000)
                    fix_input = (
                        f"{helpers.FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                        f"Current Code:\n{self.code_input}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)

                    if not generation_stopped:
                        success_fix, fixed_code = fix_code_until_success_or_limit(
                            single_file_code, self.session_folder, EVERYTHING_FILE, "initial_code", ""
                        )
                        iteration_data = {
                            'label': "Fixed Initial Code (Runtime)",
                            'code': fixed_code,
                            'output': "Fix pass completed",
                            'compile_err': "",
                        }
                        self.iteration_history.append(iteration_data)
                        self.signals.log.emit("[GUI] Runtime fix pass done.\n")

                        if success_fix:
                            initial_code_success = True
                            single_file_code = fixed_code

        if not initial_code_success and not generation_stopped:
            if not self.prompt_input.strip():
                self.signals.log.emit("[GUI] No code or prompt provided. Stopping.\n")
                self.signals.finished.emit()
                return

            write_to_file(PROMPT_FILE, self.prompt_input)
            append_to_file(EVERYTHING_FILE, f"===== Prompt =====\n{self.prompt_input}\n\n")

            combined_prompt_user = f"""
{helpers.GENERATE_PROMPT_USER}
\"\"\"{self.prompt_input}\"\"\" 
"""
            ensure_prompt_length_ok(combined_prompt_user)
            single_file_code = ai_service.call_ai(helpers.GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

            for iteration in range(1, MAX_ITERATIONS + 1):
                if generation_stopped:
                    break

                iteration_label = f"Iteration {iteration}"
                gen_code_filename = os.path.join(self.session_folder, f"generated_v{iteration}.cpp")

                success, compile_err, runtime_out = compile_run_check_code(
                    single_file_code, gen_code_filename, self.session_folder, EVERYTHING_FILE, iteration_label
                )

                iteration_data = {
                    'label': iteration_label,
                    'code': single_file_code,
                    'output': runtime_out if not compile_err else compile_err,
                    'compile_err': compile_err,
                }
                self.iteration_history.append(iteration_data)

                if compile_err:
                    self.signals.log.emit(f"[GUI] {iteration_label} Compile Error:\n{compile_err}\n\n")
                else:
                    self.signals.log.emit(f"[GUI] {iteration_label} Output:\n{runtime_out}\n\n")

                if success:
                    break
                else:
                    if compile_err:
                        self.signals.log.emit("[GUI] Attempting compile fix...\n")
                        truncated_compile_log = truncate_preserving_start_and_end(compile_err, max_length=7000)
                        fix_input = (
                            f"{helpers.FIX_PROMPT}\n"
                            f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                            f"Current Code:\n{single_file_code}"
                        )
                        ensure_prompt_length_ok(fix_input)
                        single_file_code = ai_service.call_ai(self.prompt_input, fix_input, FIX_MODEL_NAME)
                    else:
                        self.signals.log.emit("[GUI] Attempting runtime fix...\n")
                        truncated_test_output = truncate_preserving_start_and_end(runtime_out, max_length=7000)
                        fix_input = (
                            f"{helpers.FIX_PROMPT}\n"
                            f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                            f"Current Code:\n{single_file_code}"
                        )
                        ensure_prompt_length_ok(fix_input)
                        single_file_code = ai_service.call_ai(self.prompt_input, fix_input, FIX_MODEL_NAME)
            else:
                self.signals.log.emit(f"[GUI] Reached max iterations ({MAX_ITERATIONS}) without success.\n")

        self.signals.log.emit("[GUI] Process finished. Check 'History' for details.\n")
        self.signals.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Rancher 🐎")
        self.resize(1200, 800)

        # Apply the Fusion style with a dark palette
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

        # Worker thread
        self.worker_thread = None
        self._iteration_history = []

        # Connect signals
        self.run_button.clicked.connect(self.start_generation)
        self.stop_button.clicked.connect(self.stop_generation)
        self.history_list.currentRowChanged.connect(self.on_history_select)
        
        # Attach syntax highlighters
        self.cpp_highlighter = syntax_highlighter.CppSyntaxHighlighter(self.code_edit.document())
        self.output_highlighter = syntax_highlighter.OutputHighlighter(self.output_text.document())

    def on_load_from_disk(self):
        folder = QFileDialog.getExistingDirectory(self, "Select session folder to load code")
        if folder:
            code = load_last_generated_code(folder)
            if code:
                self.code_edit.setPlainText(code)
                self.append_log(f"[GUI] Loaded code from '{folder}'\n")
            else:
                self.append_log(f"[GUI] No code found in '{folder}'\n")

    def start_generation(self):
        if self.worker_thread and self.worker_thread.isRunning():
            return  # Already running

        code_input = self.code_edit.toPlainText()
        prompt_input = self.prompt_edit.toPlainText()

        # Clear old history
        self.history_list.clear()
        self._iteration_history.clear()
        self.output_text.clear()

        # Worker
        self.worker_thread = GenerationWorker(code_input, prompt_input)
        self.worker_thread.signals.log.connect(self.append_log)
        self.worker_thread.signals.finished.connect(self.generation_finished)
        self.worker_thread.started.connect(self.on_generation_started)
        self.worker_thread.start()

    def on_generation_started(self):
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_generation(self):
        global generation_stopped
        generation_stopped = True
        self.append_log("[GUI] Stop signal received. Attempting to stop generation...\n")

    def generation_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.append_log("[GUI] Generation thread finished.\n")
        if self.worker_thread:
            self._iteration_history = self.worker_thread.iteration_history
            self.history_list.clear()
            for item in self._iteration_history:
                self.history_list.addItem(item['label'])

    def append_log(self, message):
        self.output_text.moveCursor(self.output_text.textCursor().End)
        self.output_text.insertPlainText(message)
        self.output_text.moveCursor(self.output_text.textCursor().End)

    def on_history_select(self, index):
        if index < 0 or index >= len(self._iteration_history):
            return
        data = self._iteration_history[index]
        self.code_edit.setPlainText(data['code'])
        self.output_text.clear()
        if data['compile_err']:
            self.output_text.insertPlainText("[Compile Error]\n" + data['compile_err'] + "\n\n")
        self.output_text.insertPlainText(data['output'])

def gui_main_qt():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    gui_main_qt()
