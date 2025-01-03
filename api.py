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
from cpp_compiler import CppCompiler

# ------------------ Setup ------------------ #

ai_service = AIService(not ALLOW_API_CALLS)
cpp_compiler = CppCompiler()

# A simple global flag for stopping generation
generation_stopped = False

def stop_generation():
    """
    Signals that any ongoing generation process should stop
    at the next convenient point.
    """
    global generation_stopped
    generation_stopped = True

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
    request a fix from AI. Repeat until success or max iteration limit.
    """
    global generation_stopped
    current_code = initial_code

    for fix_iteration in range(1, max_iterations + 1):
        if generation_stopped:
            break

        fix_file = os.path.join(session_folder, f"{label_prefix}_fix_v{fix_iteration}.cpp")
        write_to_file(fix_file, current_code)
        append_to_file(everything_file, f"===== {label_prefix} Fix Iteration {fix_iteration} =====\n{current_code}")

        # Compile
        success, compile_output = cpp_compiler.compile_cpp(fix_file, os.path.join(session_folder, EXECUTABLE))
        iteration_output_file = os.path.join(session_folder, f"output_fix_v{fix_iteration}.txt")

        write_to_file(
            iteration_output_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Log =====\n{compile_output}\n\n"
        )
        append_to_file(
            everything_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Compile Log =====\n{compile_output}"
        )

        if not success:
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
        test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

        append_to_file(
            iteration_output_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Runtime Log =====\n{test_output}\n\n"
        )
        append_to_file(
            everything_file,
            f"===== {label_prefix} Fix Iteration {fix_iteration} Runtime Log =====\n{test_output}"
        )

        if test_success:
            return True, current_code
        else:
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
    write_to_file(code_filename, code)
    append_to_file(everything_file, f"===== {label_prefix} Code =====\n{code}")

    success, compile_output = cpp_compiler.compile_cpp(code_filename, os.path.join(session_folder, EXECUTABLE))

    output_file = os.path.join(session_folder, f"output_v{iteration_num}.txt")
    combined_log = f"===== {label_prefix} Compile Log =====\n{compile_output}\n\n"

    write_to_file(output_file, combined_log)
    append_to_file(everything_file, f"===== {label_prefix} Compile Log =====\n{compile_output}\n\n")

    if not success:
        return (False, combined_log)

    test_success, test_output = cpp_compiler.run_executable(os.path.join(session_folder, EXECUTABLE))

    runtime_log = f"===== {label_prefix} Runtime Log =====\n{test_output}\n\n"
    append_to_file(output_file, runtime_log)
    append_to_file(everything_file, runtime_log)
    combined_log += runtime_log

    if not test_success:
        return (False, combined_log)
    else:
        return (True, combined_log)

# ------------------ Load-from-disk Helpers ------------------ #

def load_iteration_history_from_folder(session_folder: str):
    """
    Loads all iteration versions from the given folder. Returns a list of dicts
    that the GUI can show in a 'History' list: [{'label':..., 'code':..., 'output':...}, ...].
    """
    if not os.path.isdir(session_folder):
        return []

    code_pattern = re.compile(r"(.*)_v(\d+)\.cpp$")
    output_pattern = re.compile(r"output.*_v(\d+)\.txt$")
    versions_map = {}

    for filename in os.listdir(session_folder):
        full_path = os.path.join(session_folder, filename)
        if os.path.isfile(full_path):
            cmatch = code_pattern.match(filename)
            if cmatch:
                prefix = cmatch.group(1)
                vstr = cmatch.group(2)
                version = int(vstr)
                if version not in versions_map:
                    versions_map[version] = {'code': "", 'output': "", 'label': f"{prefix}_v{version}"}
                versions_map[version]['code'] = read_file(full_path)
                versions_map[version]['label'] = f"{prefix}_v{version}"

            omatch = output_pattern.match(filename)
            if omatch:
                vstr = omatch.group(1)
                version = int(vstr)
                if version not in versions_map:
                    versions_map[version] = {'code': "", 'output': "", 'label': f"v{version}"}
                versions_map[version]['output'] = read_file(full_path)

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

# ------------------ High-Level Generation Logic ------------------ #

def run_generation_process(
    code_input: str,
    prompt_input: str,
    log_func=None,
    data_updated_func=None,
    finished_func=None
):
    """
    Runs the 'generation' process. 
      - code_input: user-supplied C++ code (may be blank).
      - prompt_input: user-supplied text prompt (may be blank).
      - log_func: a callback for logging messages if desired.
      - data_updated_func: a callback to let the UI know new files were written, passing the session folder.
      - finished_func: a callback to signal everything is done.
    """

    global generation_stopped
    generation_stopped = False

    # Create session folder and the everything file inside it.
    session_folder = create_session_folder(prompt_input)
    everything_file = os.path.join(session_folder, "everything.cpp")

    # Helper to conditionally log
    def log(message: str):
        if log_func:
            log_func(message)

    # Helper to alert the UI that new data is available
    def data_updated():
        if data_updated_func:
            data_updated_func(session_folder)

    log("[Logic] Starting generation process.\n")

    # PART 1: Attempt user-supplied code (if any)
    initial_code_success = False
    single_file_code = code_input.strip()

    if single_file_code and not generation_stopped:
        log("[Logic] Attempting to compile/run initial code...\n")
        init_code_filename = os.path.join(session_folder, "initial_code.cpp")
        success, combined_log = compile_run_check_code(
            single_file_code, 0, init_code_filename, session_folder, everything_file, "Initial Code"
        )
        data_updated()

        if success:
            log("[Logic] Initial code compiled and ran successfully.\n")
            initial_code_success = True
        else:
            log("[Logic] Initial code failed. Attempting AI fix...\n")
            truncated_log = truncate_preserving_start_and_end(combined_log, max_length=7000)
            fix_input = (
                f"{helpers.FIX_PROMPT}\n"
                f"Compilation/Runtime Log:\n{truncated_log}\n"
                f"Current Code:\n{single_file_code}"
            )
            ensure_prompt_length_ok(fix_input)
            log("[Logic] Requesting fix from AI...\n")
            single_file_code = ai_service.call_ai("", fix_input, FIX_MODEL_NAME)

            if not generation_stopped:
                log("[Logic] Attempting fix_code_until_success_or_limit...\n")
                success_fix, fixed_code = fix_code_until_success_or_limit(
                    single_file_code, session_folder, everything_file, "initial_code"
                )
                data_updated()

                if success_fix:
                    log("[Logic] Fixed initial code succeeded.\n")
                    initial_code_success = True
                    single_file_code = fixed_code

    # PART 2: If initial code not successful or absent, generate from prompt
    if not initial_code_success and not generation_stopped:
        if not prompt_input.strip():
            log("[Logic] No code or prompt provided. Stopping.\n")
            if finished_func:
                finished_func()
            return

        # Write prompt to disk
        prompt_file = os.path.join(session_folder, "prompt.txt")
        helpers.write_to_file(prompt_file, prompt_input)
        append_to_file(everything_file, f"===== Prompt =====\n{prompt_input}\n\n")
        data_updated()

        log("[Logic] Generating new code from prompt...\n")
        combined_prompt_user = f"""
{helpers.GENERATE_PROMPT_USER}
\"\"\"{prompt_input}\"\"\" 
"""
        ensure_prompt_length_ok(combined_prompt_user)
        single_file_code = ai_service.call_ai(helpers.GENERATE_PROMPT_SYSTEM, combined_prompt_user, INITAL_MODEL_NAME)

        for iteration in range(1, MAX_ITERATIONS + 1):
            if generation_stopped:
                break

            iteration_label = f"Iteration {iteration}"
            gen_code_filename = os.path.join(session_folder, f"generated_v{iteration}.cpp")

            log(f"[Logic] {iteration_label} - compiling/running code...\n")
            success, combined_log = compile_run_check_code(
                single_file_code,
                iteration,
                gen_code_filename,
                session_folder,
                everything_file,
                iteration_label
            )
            data_updated()

            if success:
                log(f"[Logic] {iteration_label} succeeded; stopping.\n")
                break
            else:
                truncated_log = truncate_preserving_start_and_end(combined_log, max_length=7000)
                if "error:" in combined_log.lower():
                    log("[Logic] Attempting compile fix...\n")
                    fix_input = (
                        f"{helpers.FIX_PROMPT}\n"
                        f"Compilation/Runtime Log:\n{truncated_log}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                else:
                    log("[Logic] Attempting runtime fix...\n")
                    fix_input = (
                        f"{helpers.FIX_PROMPT}\n"
                        f"Runtime/Verbose Logging Output:\n{truncated_log}\n"
                        f"Current Code:\n{single_file_code}"
                    )
                ensure_prompt_length_ok(fix_input)
                log("[Logic] Requesting fix from AI...\n")
                single_file_code = ai_service.call_ai(prompt_input, fix_input, FIX_MODEL_NAME)
        else:
            log(f"[Logic] Reached max iterations ({MAX_ITERATIONS}) without success.\n")

    log("[Logic] Process finished.\n")
    if finished_func:
        finished_func()


def create_session_folder(prompt_input):
    """
    Creates and returns a new session folder based on timestamp and prompt.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snippet = helpers.sanitize_for_filename(prompt_input)
    
    if snippet.strip():
        session_folder = os.path.join(
            GENERATED_ROOT_FOLDER,
            f"session_{timestamp}_{snippet}"
        )
    else:
        session_folder = os.path.join(
            GENERATED_ROOT_FOLDER,
            f"session_{timestamp}"
        )
    
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

if __name__ == "__main__":
    # Example usage (without GUI) would be manually calling run_generation_process(...)
    pass
