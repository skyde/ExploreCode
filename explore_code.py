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

# ------------------ Setup ------------------ #

ai_service = AIService(not ALLOW_API_CALLS)
cpp_compiler = CppCompiler()

generation_stopped = False

# ------------------ Fix / Compile / Run Helpers ------------------ #

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
    Attempt to compile and run 'initial_code'. If compilation/runtime fails,
    request a fix from OpenAI. Repeat until success or max iteration limit.
    Returns: (success: bool, final_code: str).
    """
    global generation_stopped
    current_code = initial_code

    for fix_iteration in range(1, max_iterations + 1):
        if generation_stopped:
            break

        # Save the code for this iteration
        fix_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}.cpp")
        write_to_file(fix_file, current_code)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} =====\n{current_code}")

        # Compile
        print(f"[main] Compiling {label_prefix} code iteration {fix_iteration}...\n")
        success, compile_output = cpp_compiler.compile_cpp(fix_file, os.path.join(session_folder, EXECUTABLE))

        # Single combined output file name
        iteration_output_file = os.path.join(session_folder, f"output_fix_v{fix_iteration}.txt")

        # Write the compile log
        write_to_file(
            iteration_output_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Log =====\n{compile_output}\n\n"
        )
        append_to_file(
            everything_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Log =====\n{compile_output}"
        )

        if not success:
            print("[main] Compilation failed. Requesting new fix...\n")
            truncated_compile_log = truncate_preserving_start_and_end(compile_output, max_length=7000)
            fix_input = (
                f"{fix_prompt}\n"
                f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                f"Current Code:\n{current_code}"
            )
            ensure_prompt_length_ok(fix_input)
            current_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)
            continue

        # If compilation succeeded, run
        print(f"[main] Running {label_prefix} code iteration {fix_iteration}...\n")
        test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

        # Append runtime log
        append_to_file(
            iteration_output_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Runtime Log =====\n{test_output}\n\n"
        )
        append_to_file(
            everything_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Runtime Log =====\n{test_output}"
        )

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
    iteration_num: int,
    code_filename: str,
    session_folder: str,
    everything_file: str,
    label_prefix: str
):
    """
    Compiles and runs the provided code, returning (success: bool, combined_log: str).
    The combined log merges compile + runtime logs together.
    """
    # Write the code
    write_to_file(code_filename, code)
    append_to_file(everything_file, f"===== {label_prefix} Code =====\n{code}")

    print(f"[main] Compiling {label_prefix}...\n")
    success, compile_output = cpp_compiler.compile_cpp(code_filename, os.path.join(session_folder, EXECUTABLE))

    # Single combined output file name: output_vN.txt
    output_file = os.path.join(session_folder, f"output_v{iteration_num}.txt")
    combined_log = f"===== {label_prefix} Compile Log =====\n{compile_output}\n\n"

    # Write compile log
    write_to_file(output_file, combined_log)
    append_to_file(everything_file, f"===== {label_prefix} Compile Log =====\n{compile_output}\n\n")

    if not success:
        print(f"[main] Error compiling {label_prefix}:\n{compile_output}\n")
        return (False, combined_log)  # Return compile log as the combined log

    print(f"[main] Running {label_prefix}...\n")
    test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

    # Append runtime to combined log
    runtime_log = f"===== {label_prefix} Runtime Log =====\n{test_output}\n\n"
    append_to_file(output_file, runtime_log)
    append_to_file(everything_file, runtime_log)
    combined_log += runtime_log

    if not test_success:
        print("[main] Code run failed (non-zero exit).")
        return (False, combined_log)
    else:
        print("[main] Code run succeeded (exit code = 0). Tests passed!\n")
        return (True, combined_log)


# ------------------ Load-from-disk Helpers ------------------ #

def load_iteration_history_from_folder(session_folder: str):
    """
    Loads all iteration versions from the given folder. Returns a list of dicts
    that the GUI can show in the 'History' list. Each dict has:
      {
        'label': str,   # e.g. "generated_v1"
        'code': str,    
        'output': str,  # combined compile & runtime logs
      }
    The list is sorted by version number found in the filename.
    """
    if not os.path.isdir(session_folder):
        return []

    # We look for code like "generated_vN.cpp" or "something_fix_vN.cpp"
    # and logs like "output_vN.txt" or "output_fix_vN.txt".
    code_pattern = re.compile(r"(.*)_v(\d+)\.cpp$")
    output_pattern = re.compile(r"output.*_v(\d+)\.txt$")  # or "output_v(\d+).txt"

    # Keep a map version -> { 'code': "", 'output': "", 'label': "" }
    # We can guess a label from the code or fallback to "vN".
    versions_map = {}

    for filename in os.listdir(session_folder):
        full_path = os.path.join(session_folder, filename)
        if os.path.isfile(full_path):
            # Check if it's one of the .cpp files
            cmatch = code_pattern.match(filename)
            if cmatch:
                prefix = cmatch.group(1)
                vstr = cmatch.group(2)
                version = int(vstr)
                if version not in versions_map:
                    versions_map[version] = {'code': "", 'output': "", 'label': f"{prefix}_v{version}"}
                versions_map[version]['code'] = read_file(full_path)
                versions_map[version]['label'] = f"{prefix}_v{version}"

            # Check if it's one of the output files
            omatch = output_pattern.match(filename)
            if omatch:
                vstr = omatch.group(1)
                version = int(vstr)
                if version not in versions_map:
                    versions_map[version] = {'code': "", 'output': "", 'label': f"v{version}"}
                versions_map[version]['output'] = read_file(full_path)

    # Build a list sorted by version
    iteration_list = []
    for version in sorted(versions_map.keys()):
        iteration_list.append({
            'label': versions_map[version]['label'],
            'code': versions_map[version]['code'],
            'output': versions_map[version]['output']
        })
    return iteration_list


def load_prompt_from_folder(session_folder: str) -> str:
    """
    If 'prompt.txt' exists in the session folder, return its contents;
    otherwise return an empty string.
    """
    prompt_file = os.path.join(session_folder, "prompt.txt")
    if os.path.isfile(prompt_file):
        return read_file(prompt_file)
    return ""


# ------------------ GUI Implementation ------------------ #

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

        # Try the user-supplied code first
        if self.code_input.strip() and not generation_stopped:
            iteration_label = "Initial Code"
            init_code_filename = os.path.join(self.session_folder, "initial_code.cpp")

            # We'll treat this as iteration 0 so the output is "output_v0.txt"
            success, combined_log = compile_run_check_code(
                self.code_input,
                0,
                init_code_filename,
                self.session_folder,
                EVERYTHING_FILE,
                iteration_label
            )

            # Add to iteration history
            iteration_data = {
                'label': iteration_label,
                'code': self.code_input,
                'output': combined_log
            }
            self.iteration_history.append(iteration_data)

            if success:
                initial_code_success = True
            else:
                # Attempt fix if code fails
                self.signals.log.emit(f"[GUI] {iteration_label} failed. Attempting fix...\n")
                truncated_log = truncate_preserving_start_and_end(combined_log, max_length=7000)
                fix_input = (
                    f"{helpers.FIX_PROMPT}\n"
                    f"Compilation/Runtime Log:\n{truncated_log}\n"
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
                        'output': "Fix pass completed"
                    }
                    self.iteration_history.append(iteration_data)
                    if success_fix:
                        initial_code_success = True
                        single_file_code = fixed_code

        # If initial code isn't successful, or there's no code, generate from scratch using the prompt
        if not initial_code_success and not generation_stopped:
            if not self.prompt_input.strip():
                self.signals.log.emit("[GUI] No code or prompt provided. Stopping.\n")
                self.signals.finished.emit()
                return

            # Write the prompt to the session folder
            write_to_file(PROMPT_FILE, self.prompt_input)
            append_to_file(EVERYTHING_FILE, f"===== Prompt =====\n{self.prompt_input}\n\n")

            # Ask AI to create new code
            combined_prompt_user = f"""
{helpers.GENERATE_PROMPT_USER}
\"\"\"{self.prompt_input}\"\"\" 
"""
            ensure_prompt_length_ok(combined_prompt_user)
            single_file_code = ai_service.call_ai(helpers.GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

            # Attempt compilation/fix for up to MAX_ITERATIONS
            for iteration in range(1, MAX_ITERATIONS + 1):
                if generation_stopped:
                    break

                iteration_label = f"Iteration {iteration}"
                gen_code_filename = os.path.join(self.session_folder, f"generated_v{iteration}.cpp")

                success, combined_log = compile_run_check_code(
                    single_file_code,
                    iteration,
                    gen_code_filename,
                    self.session_folder,
                    EVERYTHING_FILE,
                    iteration_label
                )

                iteration_data = {
                    'label': f"generated_v{iteration}",
                    'code': single_file_code,
                    'output': combined_log
                }
                self.iteration_history.append(iteration_data)

                if success:
                    break
                else:
                    # Attempt a fix
                    truncated_log = truncate_preserving_start_and_end(combined_log, max_length=7000)
                    if "error:" in combined_log.lower():
                        self.signals.log.emit("[GUI] Attempting compile fix...\n")
                        fix_input = (
                            f"{helpers.FIX_PROMPT}\n"
                            f"Compilation/Runtime Log:\n{truncated_log}\n"
                            f"Current Code:\n{single_file_code}"
                        )
                    else:
                        self.signals.log.emit("[GUI] Attempting runtime fix...\n")
                        fix_input = (
                            f"{helpers.FIX_PROMPT}\n"
                            f"Runtime/Verbose Logging Output:\n{truncated_log}\n"
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
            # Load iteration data (including compile & runtime logs)
            loaded_history = load_iteration_history_from_folder(folder)
            # Also load prompt
            loaded_prompt = load_prompt_from_folder(folder)

            # Clear UI
            self.history_list.clear()
            self._iteration_history.clear()
            self.output_text.clear()

            if loaded_prompt:
                self.prompt_edit.setPlainText(loaded_prompt)

            if loaded_history:
                self._iteration_history = loaded_history
                for item in self._iteration_history:
                    self.history_list.addItem(item['label'])

                # Automatically select & display the last iteration
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
        """
        When the user clicks on an iteration in the History list,
        show that iteration's code and combined logs.
        """
        if index < 0 or index >= len(self._iteration_history):
            return

        data = self._iteration_history[index]
        self.code_edit.setPlainText(data['code'])
        self.output_text.clear()
        self.output_text.insertPlainText(data['output'])


def gui_main_qt():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui_main_qt()
