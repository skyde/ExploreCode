#!/usr/bin/env python3

import os
import sys
import subprocess
import datetime
from dotenv import load_dotenv
from openai import OpenAI
import helpers

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
    QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QRegExp
from PyQt5.QtGui import (
    QPalette, QColor, QSyntaxHighlighter, QTextCharFormat, QFont
)
import queue

# ------------------ Syntax Highlighting Classes ------------------ #

class CppSyntaxHighlighter(QSyntaxHighlighter):
    """
    A simple C++ syntax highlighter for QPlainTextEdit.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlight_rules = []

        # Create text formats
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#C586C0"))  # Purple hue
        keyword_format.setFontWeight(QFont.Bold)

        # C++ keywords
        keywords = [
            "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit", "atomic_noexcept",
            "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
            "class", "compl", "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
            "explicit", "export", "extern", "false", "final", "float", "for", "friend", "goto", "if", "inline", "int",
            "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
            "private", "protected", "public", "reflexpr", "register", "reinterpret_cast", "requires", "return",
            "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "synchronized",
            "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union",
            "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
        ]

        # Build rules for each keyword
        for word in keywords:
            pattern = QRegExp(r"\b" + word + r"\b")
            self._highlight_rules.append((pattern, keyword_format))

        # Class name format
        class_format = QTextCharFormat()
        class_format.setForeground(QColor("#4EC9B0"))  # Light green
        pattern = QRegExp(r"\bQ[A-Z]\w*\b")
        self._highlight_rules.append((pattern, class_format))

        # Single-line comment format
        single_line_comment_format = QTextCharFormat()
        single_line_comment_format.setForeground(QColor("#608B4E"))
        self._highlight_rules.append((QRegExp(r"//[^\n]*"), single_line_comment_format))

        # Multi-line comment format
        self._multi_line_comment_format = QTextCharFormat()
        self._multi_line_comment_format.setForeground(QColor("#608B4E"))

        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))
        self._highlight_rules.append((QRegExp(r"\".*\""), string_format))
        self._highlight_rules.append((QRegExp(r"\'.*\'"), string_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))
        self._highlight_rules.append((QRegExp(r"\b[0-9]+(\.[0-9]+)?\b"), number_format))

        # Multi-line comment delimiters
        self._comment_start = QRegExp(r"/\*")
        self._comment_end = QRegExp(r"\*/")

    def highlightBlock(self, text):
        # Apply standard rules
        for pattern, fmt in self._highlight_rules:
            index = pattern.indexIn(text, 0)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)

        # Handle multi-line comments
        self.setCurrentBlockState(0)

        start_index = 0
        if self.previousBlockState() != 1:
            start_index = self._comment_start.indexIn(text)

        while start_index >= 0:
            end_index = self._comment_end.indexIn(text, start_index)
            if end_index == -1:
                self.setCurrentBlockState(1)
                comment_length = len(text) - start_index
            else:
                comment_length = end_index - start_index + self._comment_end.matchedLength()
            self.setFormat(start_index, comment_length, self._multi_line_comment_format)
            if end_index == -1:
                break
            else:
                start_index = self._comment_start.indexIn(text, start_index + comment_length)


class OutputHighlighter(QSyntaxHighlighter):
    """
    A simple highlighter for the output log. 
    Highlights lines containing 'error', 'warning', or 'info'.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def highlightBlock(self, text):
        lower_text = text.lower()
        if "error" in lower_text:
            error_format = QTextCharFormat()
            error_format.setForeground(QColor("#F44747"))  # Red
            self.setFormat(0, len(text), error_format)
        elif "warning" in lower_text:
            warn_format = QTextCharFormat()
            warn_format.setForeground(QColor("#FFA500"))  # Orange
            self.setFormat(0, len(text), warn_format)
        elif "info" in lower_text:
            info_format = QTextCharFormat()
            info_format.setForeground(QColor("#0FAF0F"))  # Green
            self.setFormat(0, len(text), info_format)

# ------------------ Original Configuration ------------------ #

DEBUG_PROMPT = """
- Generate an algorithm that calculates connectivity clustering given a graph of nodes and edges
- Each node is connected to 1-8 other nodes
- The algorithm should be very efficient, and create benchmarks to verify performance
- Use a 'bit matrix' to accelerate the connectivity clustering, using 256 bit registers, and clever use of intrinsics and bit manipulation operations to accelerate (e.g. bitwise OR)
- Use SIMD operations (AVX2)
- Only use a single core
"""
USE_DEBUG_PROMPT = False
INITAL_MODEL_NAME = "o1-preview"
FIX_MODEL_NAME = "o1-mini"
MAX_ITERATIONS = 10

GENERATED_ROOT_FOLDER = "generated"
EXECUTABLE = "program"  # We'll compile the code to a program each iteration
PRINT_SEND = True

GENERATE_PROMPT_SYSTEM = r"""
Write a single C++17 file which I'll call 'generated.cpp'. 
Your solution must:
- Include a main() function that unit tests the relevant functionality using <cassert>.
- Tests should be EXTREMELY extensive and cover all edge cases.
- Add verbose logging to the tests as they run so we can see what's happening from the output. You should output the values of things as they run so we can see what they are.
- If any test fails, the program should exit with a non-zero return code.
- Reply with ONLY code. Your response will be directly fed into the Clang compiler, so anything else will result in a compilation error.
- Do NOT put code in 
cpp blocks.
"""

GENERATE_PROMPT_USER = r"""
Please solve the following problem:
"""

FIX_PROMPT = r"""
We encountered errors during compilation or runtime.
Please fix the entire single-file code.
Add additional verbose logging to the tests as the previous logging was insufficient. You should output the values of things as they run so we can see what they are.
Be careful about using verbose logging in tight loops, to avoid excessive output.
Reply with ONLY code. Your response will be directly fed into the Clang compiler, so anything else will result in a compilation error.
Do NOT put code in
cpp blocks.
"""

MAX_PROMPT_LENGTH = 50000
DEBUG_FIX_CODE = helpers.DEBUG_VALID_CODE

# ------------------ Setup OpenAI ------------------ #

load_dotenv()  # so OPENAI_API_KEY is available
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

# ------------------ OpenAI Helpers ------------------ #

def call_openai(system_prompt, user_prompt, model, temperature=0.2):
    """
    Calls the OpenAI client with a system prompt and user prompt, returning the response text.
    """
    if PRINT_SEND:
        print("[call_openai] Sending prompt to OpenAI:")
        print("------------------ System --------------")
        print(system_prompt)
        print("------------------ User ----------------\n")
        print(user_prompt)
        print("----------------------------------------\n")

    # For non-supported models, just use the user role
    system_role = "user"

    response = client.chat.completions.create(
        messages=[{"role": system_role, "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model
    )

    answer = response.choices[0].message.content

    print("[call_openai] Received response from OpenAI:")
    print("--------------------------------------------")
    print(answer)
    print("--------------------------------------------\n")

    return answer

# ------------------ File Helpers ------------------ #

def write_to_file(filename, content):
    """Overwrites the file with the given content."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"[write_to_file] Writing to {filename}...\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

def append_to_file(filename, content):
    """Appends the given content to the file with a separator line."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n")

def read_file(filename):
    """Reads and returns the content of a file."""
    print(f"[read_file] Reading from {filename}...\n")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# ------------------ Compilation & Execution Helpers ------------------ #

def compile_cpp(source_file, output_file):
    """
    Compiles the C++ file into an executable using clang++.
    Returns (success, error_message).
    """
    compiler = "clang++"
    print(f"[compile_cpp] Using compiler: {compiler}")
    
    cmd = [compiler, "-std=c++17", "-mavx2", source_file, "-o", output_file]
    print(f"[compile_cpp] Command: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        return (process.returncode == 0, stderr.decode("utf-8"))
    except FileNotFoundError:
        return (False, f"Compiler '{compiler}' not found. Ensure it is installed and on your PATH.")

def run_executable(executable):
    """
    Runs the executable file and returns (success, combined_output).
    """
    print(f"[run_executable] Running ./{executable}...\n")
    cmd = [f"./{executable}"]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate(timeout=30)
        return (process.returncode == 0, out.decode("utf-8") + "\n" + err.decode("utf-8"))
    except subprocess.TimeoutExpired:
        process.kill()
        return (False, "Timeout expired while running the code. Possibly an infinite loop.\n")
    except FileNotFoundError:
        return (False, f"Executable '{executable}' not found. Ensure it was created successfully.\n")

# ------------------ Fix / Compile / Run Helpers ------------------ #

generation_stopped = False

def fix_code_until_success_or_limit(
    initial_code: str,
    session_folder: str,
    everything_file: str,
    label_prefix: str,
    user_problem: str = "",
    fix_prompt: str = FIX_PROMPT,
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
        success, error_message = compile_cpp(fix_file, os.path.join(session_folder, EXECUTABLE))

        compile_output_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}_compile.txt")
        write_to_file(compile_output_file, error_message)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Output =====\n{error_message}")

        if not success:
            print("[main] Compilation failed. Requesting new fix...\n")
            truncated_compile_log = maybe_truncate_for_llm(error_message, max_length=7000)
            fix_input = (
                f"{fix_prompt}\n"
                f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                f"Current Code:\n{current_code}"
            )
            ensure_prompt_length_ok(fix_input)
            current_code = call_openai("", fix_input, FIX_MODEL_NAME)
            continue

        print(f"[main] Running {label_prefix} code iteration {fix_iteration}...\n")
        test_success, test_output = run_executable(os.path.join(session_folder, EXECUTABLE))

        output_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}_output.txt")
        write_to_file(output_file, test_output)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} Run Output =====\n{test_output}")

        if test_success:
            print(f"[main] {label_prefix} code now passes tests!\n")
            return True, current_code
        else:
            print("[main] Code failed again. Requesting further fix...\n")
            truncated_test_output = maybe_truncate_for_llm(test_output, max_length=7000)
            fix_input = (
                f"{fix_prompt}\n"
                f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                f"Current Code:\n{current_code}"
            )
            ensure_prompt_length_ok(fix_input)
            current_code = call_openai("", fix_input, FIX_MODEL_NAME)

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
    success, error_message = compile_cpp(code_filename, os.path.join(session_folder, EXECUTABLE))

    compile_output_file = os.path.join(session_folder, f"{label_prefix}_compile.txt")
    write_to_file(compile_output_file, error_message)
    append_to_file(everything_file, f"===== {label_prefix} Compile Output =====\n{error_message}")

    if not success:
        print(f"[main] Error compiling {label_prefix}:\n{error_message}\n")
        return (False, error_message, "")

    print(f"[main] Running {label_prefix}...\n")
    test_success, test_output = run_executable(os.path.join(session_folder, EXECUTABLE))

    output_file = os.path.join(session_folder, f"{label_prefix}_output.txt")
    write_to_file(output_file, test_output)
    append_to_file(everything_file, f"===== {label_prefix} Run Output =====\n{test_output}")

    if not test_success:
        print("[main] Code run failed (non-zero exit).")
        return (False, "", test_output)
    else:
        print("[main] Code run succeeded (exit code = 0). Tests passed!\n")
        return (True, "", test_output)

# ------------------ Original CLI Main ------------------ #

def cli_main():
    print("[main] Starting code generation & testing process.\n")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    EVERYTHING_FILE = os.path.join(session_folder, "everything.cpp")
    PROMPT_FILE = os.path.join(session_folder, "prompt.txt")

    print("[main] Please enter your initial code if needed (Ctrl+Z/Win or Ctrl+D/*nix to finish):\n")
    initial_code = sys.stdin.read()

    initial_code_success = False
    single_file_code = ""

    if initial_code.strip():
        print("[main] We have initial code from stdin. We'll try compiling and running it.\n")
        init_code_filename = os.path.join(session_folder, "initial_code.cpp")
        success, compile_err, runtime_out = compile_run_check_code(
            initial_code, init_code_filename, session_folder, EVERYTHING_FILE, "Initial Code"
        )

        if success:
            initial_code_success = True
        else:
            if compile_err:
                truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                fix_input = (
                    f"{FIX_PROMPT}\n"
                    f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                    f"Current Code:\n{initial_code}"
                )
                ensure_prompt_length_ok(fix_input)
                single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)
                initial_code_success, single_file_code = fix_code_until_success_or_limit(
                    single_file_code, session_folder, EVERYTHING_FILE, "initial_code", ""
                )
            else:
                truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                fix_input = (
                    f"{FIX_PROMPT}\n"
                    f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                    f"Current Code:\n{initial_code}"
                )
                ensure_prompt_length_ok(fix_input)
                single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)
                initial_code_success, single_file_code = fix_code_until_success_or_limit(
                    single_file_code, session_folder, EVERYTHING_FILE, "initial_code", ""
                )

        if initial_code_success:
            print("[main] The code from stdin worked or was fixed.\n")

    if not initial_code_success:
        print("[main] Proceeding to generate new code.\n")

        if USE_DEBUG_PROMPT:
            user_problem = DEBUG_PROMPT
        else:
            user_problem = input("Enter a short description of the problem you want solved: ")

        write_to_file(PROMPT_FILE, user_problem)
        append_to_file(EVERYTHING_FILE, f"===== Prompt =====\n{user_problem}\n\n")

        combined_prompt_user = f"""
{GENERATE_PROMPT_USER}
\"\"\"{user_problem}\"\"\" 
"""
        ensure_prompt_length_ok(combined_prompt_user)
        single_file_code = call_openai(GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

        for iteration in range(1, MAX_ITERATIONS + 1):
            gen_code_filename = os.path.join(session_folder, f"generated_v{iteration}.cpp")
            success, compile_err, runtime_out = compile_run_check_code(
                single_file_code, gen_code_filename, session_folder, EVERYTHING_FILE, f"Iteration {iteration}"
            )
            if success:
                break
            else:
                if compile_err:
                    truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai(user_problem, fix_input, FIX_MODEL_NAME)
                else:
                    truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai(user_problem, fix_input, FIX_MODEL_NAME)
        else:
            print(f"[main] Reached the maximum number of iterations ({MAX_ITERATIONS}) without success.\n")

    print("[main] Process finished.\n")
    print(f"[main] Code & logs in '{session_folder}'. Combined log in '{EVERYTHING_FILE}'.\n")

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
        self.session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_qt_{timestamp}")
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
                    truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                        f"Current Code:\n{self.code_input}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)

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
                    truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                        f"Current Code:\n{self.code_input}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)

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
{GENERATE_PROMPT_USER}
\"\"\"{self.prompt_input}\"\"\" 
"""
            ensure_prompt_length_ok(combined_prompt_user)
            single_file_code = call_openai(GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

            for iteration in range(1, MAX_ITERATIONS + 1):
                if generation_stopped:
                    break

                iteration_label = f"Generated Iteration {iteration}"
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
                        truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                        fix_input = (
                            f"{FIX_PROMPT}\n"
                            f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                            f"Current Code:\n{single_file_code}"
                        )
                        ensure_prompt_length_ok(fix_input)
                        single_file_code = call_openai(self.prompt_input, fix_input, FIX_MODEL_NAME)
                    else:
                        self.signals.log.emit("[GUI] Attempting runtime fix...\n")
                        truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                        fix_input = (
                            f"{FIX_PROMPT}\n"
                            f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                            f"Current Code:\n{single_file_code}"
                        )
                        ensure_prompt_length_ok(fix_input)
                        single_file_code = call_openai(self.prompt_input, fix_input, FIX_MODEL_NAME)
            else:
                self.signals.log.emit(f"[GUI] Reached max iterations ({MAX_ITERATIONS}) without success.\n")

        self.signals.log.emit("[GUI] Process finished. Check 'History' for details.\n")
        self.signals.finished.emit()


from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QSplitter
)
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("C++ Generation & Fixer (PyQt Dark Mode)")
        self.resize(1200, 800)

        # Apply the Fusion style with a dark palette
        self.apply_dark_mode()

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Prompt area
        self.prompt_label = QLabel("Prompt (multi-line):")
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
        self.history_label = QLabel("Past Generations:")
        self.history_list = QListWidget()
        left_layout.addWidget(self.history_label)
        left_layout.addWidget(self.history_list)
        splitter_main.addWidget(left_widget)

        # Right -> Another splitter
        splitter_right = QSplitter(Qt.Vertical)

        # Output
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        self.output_label = QLabel("Output (Compile/Error/Run Logs):")
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_text)
        output_widget.setLayout(output_layout)

        splitter_right.addWidget(output_widget)

        # Code
        code_widget = QWidget()
        code_layout = QVBoxLayout(code_widget)
        self.code_label = QLabel("Paste your C++ code (optional):")
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
        self.cpp_highlighter = CppSyntaxHighlighter(self.code_edit.document())
        self.output_highlighter = OutputHighlighter(self.output_text.document())

    def apply_dark_mode(self):
        """Apply a dark palette to the entire application."""
        app = QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(dark_palette)

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
            # Move iteration_history from worker to local
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
    # If user passes 'cli' argument, run the CLI version. Otherwise, run the PyQt (Dark Mode) GUI.
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        cli_main()
    else:
        gui_main_qt()
