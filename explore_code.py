#!/usr/bin/env python3

import os
import sys
import subprocess
import datetime
from dotenv import load_dotenv
from openai import OpenAI
import helpers
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue

# ------------------ Configuration ------------------ #

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
    compiler = "clang++"  # Use clang++ instead of g++
    print(f"[compile_cpp] Using compiler: {compiler}")
    
    # Compile the C++ file with C++17 standard and AVX2
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
    Includes a timeout to avoid infinite loops.
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

generation_stopped = False  # Global flag to allow stopping generation mid-way

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

    Returns (success, final_code).
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
    The code is saved to code_filename and appended to everything_file.
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

# ------------------ CLI Main (Original Process) ------------------ #

def cli_main():
    """
    This is the original 'main()' function from the code.
    It reads code from stdin, tries to compile & run it, or else
    generates new code from a prompt. All logic remains unchanged.
    """
    print("[main] Starting code generation & testing process.\n")

    # Create a new subfolder in "generated/" with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    EVERYTHING_FILE = os.path.join(session_folder, "everything.cpp")
    PROMPT_FILE = os.path.join(session_folder, "prompt.txt")

    # ------------------ Read code from stdin ------------------ #
    print("[main] Please enter your initial code if needed (Ctrl+Z on Windows or Ctrl+D on Linux/Mac to finish):\n")
    initial_code = sys.stdin.read()

    initial_code_success = False
    single_file_code = ""

    # 2) If we got any code from stdin, compile and run it
    if initial_code.strip():
        print("[main] We have initial code from stdin. We'll try compiling and running it.\n")
        initial_code_filename = os.path.join(session_folder, "initial_code.cpp")
        success, compile_err, runtime_out = compile_run_check_code(
            initial_code, initial_code_filename, session_folder, EVERYTHING_FILE, "Initial Code"
        )

        if success:
            initial_code_success = True
        else:
            # If it's a compile error
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
                # It's a runtime error
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
            print("[main] The code from stdin worked or was fixed to work. No further generation needed.\n")

    # 3) If the initial code did NOT succeed (or nothing was provided), generate new code from a user prompt
    if not initial_code_success:
        print("[main] Proceeding to generate new code.\n")

        # If in debug mode, use DEBUG_PROMPT
        if USE_DEBUG_PROMPT:
            user_problem = DEBUG_PROMPT
        else:
            user_problem = input("Enter a short description of the problem you want solved: ")

        write_to_file(PROMPT_FILE, user_problem)
        append_to_file(EVERYTHING_FILE, f"===== Prompt =====\n{user_problem}\n\n")

        # Generate brand new single-file code
        combined_prompt_user = f"""
{GENERATE_PROMPT_USER}
\"\"\"{user_problem}\"\"\" 
"""
        ensure_prompt_length_ok(combined_prompt_user)
        single_file_code = call_openai(GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

        # Attempt compile & run, else fix in a loop
        for iteration in range(1, MAX_ITERATIONS + 1):
            generated_code_filename = os.path.join(session_folder, f"generated_v{iteration}.cpp")
            success, compile_err, runtime_out = compile_run_check_code(
                single_file_code, generated_code_filename, session_folder, EVERYTHING_FILE, f"Iteration {iteration}"
            )

            if success:
                break  # success
            else:
                # If it's a compile error
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
                    # It's a runtime error
                    truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai(user_problem, fix_input, FIX_MODEL_NAME)
        else:
            print(f"[main] Reached the maximum number of iterations ({MAX_ITERATIONS}) without passing tests.\n")

    print("[main] Process finished.\n")
    print(f"[main] All generated code and outputs have been saved in the '{session_folder}' folder.")
    print(f"[main] A combined log of everything can be found in '{EVERYTHING_FILE}'.\n")

# ------------------ GUI Enhancements ------------------ #

iteration_history = []
log_queue = queue.Queue()
worker_thread = None

def stop_generation():
    """
    Sets the global flag so that the generation loop can stop gracefully.
    """
    global generation_stopped
    generation_stopped = True
    log_queue.put("[GUI] Stop signal received. Attempting to stop generation...\n")

def log_message(msg: str):
    """
    Thread-safe way to add text to the queue, which the main thread will display.
    """
    log_queue.put(msg)

def process_log_queue(output_text_widget):
    """
    Periodically called in the main thread to consume messages from the log_queue
    and display them in the output widget.
    """
    try:
        while True:
            msg = log_queue.get_nowait()
            output_text_widget.insert(tk.END, msg)
            output_text_widget.see(tk.END)
    except queue.Empty:
        pass
    # Schedule next check
    output_text_widget.after(100, lambda: process_log_queue(output_text_widget))

def run_generation_gui(code_input, user_prompt, output_text_widget, history_list, root):
    """
    Encapsulates the generation/fix loop using the user-provided code and prompt,
    storing iteration info in `iteration_history`. Checks 'generation_stopped' to stop early.
    """
    global generation_stopped
    generation_stopped = False

    log_message("[GUI] Starting generation process...\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(GENERATED_ROOT_FOLDER, f"session_gui_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    EVERYTHING_FILE = os.path.join(session_folder, "everything.cpp")
    PROMPT_FILE = os.path.join(session_folder, "prompt.txt")

    initial_code_success = False
    single_file_code = ""

    # 1) Attempt with user-provided code_input first
    if code_input.strip() and not generation_stopped:
        iteration_label = "Initial Code"
        code_filename = os.path.join(session_folder, "initial_code.cpp")

        success, compile_err, runtime_out = compile_run_check_code(
            code_input, code_filename, session_folder, EVERYTHING_FILE, iteration_label
        )

        iteration_data = {
            'label': iteration_label,
            'code': code_input,
            'output': runtime_out if not compile_err else compile_err,
            'compile_err': compile_err,
        }
        iteration_history.append(iteration_data)
        history_list.insert(tk.END, iteration_label)

        if compile_err:
            log_message(f"[GUI] {iteration_label} Compile Error:\n{compile_err}\n\n")
        else:
            log_message(f"[GUI] {iteration_label} Output:\n{runtime_out}\n\n")

        if success:
            initial_code_success = True
        else:
            if compile_err:
                log_message("[GUI] Fixing compile error...\n")
                truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                fix_input = (
                    f"{FIX_PROMPT}\n"
                    f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                    f"Current Code:\n{code_input}"
                )
                ensure_prompt_length_ok(fix_input)
                single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)

                if not generation_stopped:
                    success_fix, fixed_code = fix_code_until_success_or_limit(
                        single_file_code, session_folder, EVERYTHING_FILE, "initial_code", ""
                    )
                    iteration_data = {
                        'label': "Fixed Initial Code",
                        'code': fixed_code,
                        'output': "Fix pass completed",
                        'compile_err': "",
                    }
                    iteration_history.append(iteration_data)
                    history_list.insert(tk.END, iteration_data['label'])
                    log_message("[GUI] Compile fix pass done.\n")

                    if success_fix:
                        initial_code_success = True
                        single_file_code = fixed_code

            else:
                # It's a runtime error
                log_message("[GUI] Fixing runtime error...\n")
                truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                fix_input = (
                    f"{FIX_PROMPT}\n"
                    f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                    f"Current Code:\n{code_input}"
                )
                ensure_prompt_length_ok(fix_input)
                single_file_code = call_openai("", fix_input, FIX_MODEL_NAME)

                if not generation_stopped:
                    success_fix, fixed_code = fix_code_until_success_or_limit(
                        single_file_code, session_folder, EVERYTHING_FILE, "initial_code", ""
                    )
                    iteration_data = {
                        'label': "Fixed Initial Code (Runtime)",
                        'code': fixed_code,
                        'output': "Fix pass completed",
                        'compile_err': "",
                    }
                    iteration_history.append(iteration_data)
                    history_list.insert(tk.END, iteration_data['label'])
                    log_message("[GUI] Runtime fix pass done.\n")

                    if success_fix:
                        initial_code_success = True
                        single_file_code = fixed_code

    # 2) If code not provided or still not success, attempt generation from user_prompt
    if not initial_code_success and not generation_stopped:
        if not user_prompt.strip():
            log_message("[GUI] No code or prompt provided. Stopping.\n")
            return

        write_to_file(PROMPT_FILE, user_prompt)
        append_to_file(EVERYTHING_FILE, f"===== Prompt =====\n{user_prompt}\n\n")

        combined_prompt_user = f"""
{GENERATE_PROMPT_USER}
\"\"\"{user_prompt}\"\"\" 
"""
        ensure_prompt_length_ok(combined_prompt_user)
        single_file_code = call_openai(GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

        for iteration in range(1, MAX_ITERATIONS + 1):
            if generation_stopped:
                break

            iteration_label = f"Generated Iteration {iteration}"
            generated_code_filename = os.path.join(session_folder, f"generated_v{iteration}.cpp")

            success, compile_err, runtime_out = compile_run_check_code(
                single_file_code,
                generated_code_filename,
                session_folder,
                EVERYTHING_FILE,
                iteration_label
            )

            iteration_data = {
                'label': iteration_label,
                'code': single_file_code,
                'output': runtime_out if not compile_err else compile_err,
                'compile_err': compile_err,
            }
            iteration_history.append(iteration_data)
            history_list.insert(tk.END, iteration_label)

            if compile_err:
                log_message(f"[GUI] {iteration_label} Compile Error:\n{compile_err}\n\n")
            else:
                log_message(f"[GUI] {iteration_label} Output:\n{runtime_out}\n\n")

            if success:
                break
            else:
                if compile_err:
                    log_message("[GUI] Attempting compile fix...\n")
                    truncated_compile_log = maybe_truncate_for_llm(compile_err, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Compilation/Verbose Logging Output:\n{truncated_compile_log}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai(user_prompt, fix_input, FIX_MODEL_NAME)
                else:
                    log_message("[GUI] Attempting runtime fix...\n")
                    truncated_test_output = maybe_truncate_for_llm(runtime_out, max_length=7000)
                    fix_input = (
                        f"{FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_test_output}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                    ensure_prompt_length_ok(fix_input)
                    single_file_code = call_openai(user_prompt, fix_input, FIX_MODEL_NAME)
        else:
            log_message(f"[GUI] Reached max iterations ({MAX_ITERATIONS}) without success.\n")

    log_message("[GUI] Process finished. Check 'History' for details.\n")

def background_generation(code_text, prompt_text, output_text, history_list, run_btn, stop_btn, root):
    """
    Runs the generation in a background thread, so the GUI remains responsive.
    """
    # Disable run button while generating
    run_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)

    code_input = code_text.get("1.0", tk.END)
    user_prompt = prompt_text.get("1.0", tk.END)

    # Clear iteration history for a fresh run
    iteration_history.clear()
    history_list.delete(0, tk.END)

    run_generation_gui(code_input, user_prompt, output_text, history_list, root)

    # Re-enable run button after completion
    run_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)

def on_run_button_click(code_text, prompt_text, output_text, history_list, run_btn, stop_btn, root):
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    worker_thread = threading.Thread(
        target=background_generation, 
        args=(code_text, prompt_text, output_text, history_list, run_btn, stop_btn, root),
        daemon=True
    )
    worker_thread.start()

def on_history_select(evt, code_text, output_text, history_list):
    """
    When the user selects an item in the history listbox, show the code & output for that iteration.
    """
    w = evt.widget
    if not w.curselection():
        return
    index = int(w.curselection()[0])
    data = iteration_history[index]

    code_text.delete("1.0", tk.END)
    code_text.insert(tk.END, data['code'])

    output_text.delete("1.0", tk.END)
    if data['compile_err']:
        output_text.insert(tk.END, "[Compile Error]\n" + data['compile_err'] + "\n\n")
    output_text.insert(tk.END, data['output'])

def gui_main():
    """
    Launches the Tkinter GUI with a PanedWindow-based layout:
      - Top: Prompt + Run/Stop Buttons
      - Middle: PanedWindow 
         * Left: History
         * Right: Another PanedWindow with Output (top) & Code (bottom)
    """
    root = tk.Tk()
    root.title("C++ Generation & Fixer GUI (Dark Mode, PanedWindow)")

    # ------------------ Dark Theme Setup ------------------ #
    style = ttk.Style()
    style.theme_use('clam')
    style.configure(".", background="gray15", foreground="white")
    style.configure("TFrame", background="gray15", foreground="white")
    style.configure("TLabel", background="gray15", foreground="white")
    style.configure("TButton", background="gray25", foreground="white")
    style.configure("TScrollbar", background="gray25")

    root.configure(bg="gray15")
    root.geometry("1200x800")  # A decent initial size

    # Top Frame for prompt + run/stop
    top_frame = ttk.Frame(root, padding="5")
    top_frame.pack(side=tk.TOP, fill=tk.X)

    prompt_label = ttk.Label(top_frame, text="Prompt (multi-line):")
    prompt_label.pack(side=tk.TOP, anchor=tk.W)

    prompt_text = scrolledtext.ScrolledText(top_frame, wrap=tk.WORD, height=5)
    prompt_text.pack(side=tk.TOP, fill=tk.X, expand=False)
    prompt_text.config(bg="gray20", fg="white", insertbackground="white")

    button_frame = ttk.Frame(top_frame)
    button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    run_button = ttk.Button(button_frame, text="Run Generation")
    run_button.pack(side=tk.LEFT, padx=5)

    stop_button = ttk.Button(button_frame, text="Stop Generation", command=stop_generation)
    stop_button.pack(side=tk.LEFT, padx=5)
    stop_button.config(state=tk.DISABLED)

    # Main Paned Window
    main_pane = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Left Frame: History
    left_frame = ttk.Frame(main_pane)
    left_frame.columnconfigure(0, weight=1)
    left_frame.rowconfigure(1, weight=1)

    history_label = ttk.Label(left_frame, text="Past Generations / Fixes:")
    history_label.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    history_list = tk.Listbox(left_frame, width=30, bg="gray20", fg="white", selectbackground="gray40")
    history_list.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    history_list.bind('<<ListboxSelect>>', lambda evt: on_history_select(evt, code_text, output_text, history_list))

    main_pane.add(left_frame, weight=1)

    # Right Pane: Split vertically for Output (top) & Code (bottom)
    right_pane = ttk.Panedwindow(main_pane, orient=tk.VERTICAL)
    main_pane.add(right_pane, weight=4)

    # Output Panel
    output_frame = ttk.Frame(right_pane)
    output_frame.columnconfigure(0, weight=1)
    output_frame.rowconfigure(1, weight=1)

    output_label = ttk.Label(output_frame, text="Output (Compile/Error/Run Logs):")
    output_label.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD)
    output_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    output_text.config(bg="gray20", fg="white", insertbackground="white")

    right_pane.add(output_frame, weight=2)

    # Code Panel
    code_frame = ttk.Frame(right_pane)
    code_frame.columnconfigure(0, weight=1)
    code_frame.rowconfigure(1, weight=1)

    code_label = ttk.Label(code_frame, text="Paste your C++ code (optional):")
    code_label.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    code_text = scrolledtext.ScrolledText(code_frame, wrap=tk.WORD)
    code_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    code_text.config(bg="gray20", fg="white", insertbackground="white")

    right_pane.add(code_frame, weight=3)

    # Wire up the run button to the background thread
    run_button.config(
        command=lambda: on_run_button_click(code_text, prompt_text, output_text, history_list, run_button, stop_button, root)
    )

    # Start processing the log queue for real-time updates
    process_log_queue(output_text)

    root.mainloop()

# ------------------ Entry Point ------------------ #

if __name__ == "__main__":
    # If user passes 'cli' argument, run the CLI version. Otherwise, run the GUI.
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        cli_main()
    else:
        gui_main()
