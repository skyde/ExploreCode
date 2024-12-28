import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# ------------------ Configuration ------------------ #

# DEBUG_PROMPT = "SIMD struct of arrays class that uses SIMD operations (AVX2)"
DEBUG_PROMPT = """
Ball and point based simulation
Points repel each other
Connected by springs
SIMD
"""
DEBUG_MODE = False  # Toggle this to True for local debugging without API calls
# DEBUG_MODE = True 
# MODEL_NAME = "gpt-4o-mini"
MODEL_NAME = "o1-mini"
MAX_ITERATIONS = 4

# We'll now generate per-iteration files: generated_v1.cpp, generated_v2.cpp, etc.
# as well as corresponding output logs: generated_v1_output.txt, etc.
# and a combined file everything.txt in the generated/ subfolder
GENERATED_FOLDER = "generated"
EVERYTHING_FILE = os.path.join(GENERATED_FOLDER, "everything.cpp")

EXECUTABLE = "program"  # We'll compile the code to a program each iteration
PRINT_SEND = True

GENERATE_PROMPT = r"""
Write a single C++17 file which I'll call 'generated.cpp'. 
Your solution must:
- Include a main() function that unit tests the relevant functionality using <cassert>.
- Tests should be EXTREMELY extensive and cover all edge cases.
- Add verbose logging to the tests as they run so we can see what's happening from the output. You should output the values of things as they run so we can see what they are.
- If any test fails, the program should exit with a non-zero return code.
- Reply with ONLY code. Your response will be directly fed into the Clang compiler, so anything else will result in a compilation error.
- Do NOT put code in ```cpp blocks.

Please solve the following problem:
"""

FIX_PROMPT = r"""
We encountered errors during compilation or runtime.
Please fix the entire single-file code.
Add additional verbose logging to the tests as the previous logging was insufficient. You should output the values of things as they run so we can see what they are.
Be careful about using verbose logging in tight loops, to avoid excessive output.
Reply with ONLY code. Your response will be directly fed into the Clang compiler, so anything else will result in a compilation error.
Do NOT put code in ```cpp blocks.
"""

# We will crash if prompts exceed this length
MAX_PROMPT_LENGTH = 25000

# ------------------ Debug Mode Hard-Coded Responses ------------------ #

# A simple valid C++ file
DEBUG_VALID_CODE = r"""
#include <iostream>
#include <cassert>
#include <array>
#include <algorithm>
#include <numeric>

template <typename T, std::size_t N>
class SIMDStructOfArrays {
public:
    SIMDStructOfArrays() = default;

    void set(std::size_t index, const std::array<T, N>& values) {
        assert(index < N);
        data[index] = values;
    }

    std::array<T, N> get(std::size_t index) const {
        assert(index < N);
        return data[index];
    }

    std::array<T, N> sum() const {
        std::array<T, N> result{};
        for (const auto& arr : data) {
            for (std::size_t i = 0; i < N; ++i) {
                result[i] += arr[i];
            }
        }
        return result;
    }

    std::array<T, N> multiply(const std::array<T, N>& factors) const {
        std::array<T, N> result{};
        for (const auto& arr : data) {
            for (std::size_t i = 0; i < N; ++i) {
                result[i] += arr[i] * factors[i];
            }
        }
        return result;
    }

    void clear() {
        for (auto& arr : data) {
            arr.fill(T{});
        }
    }

private:
    std::array<std::array<T, N>, N> data{};
};

void run_tests() {
    SIMDStructOfArrays<int, 3> simd;

    // Test setting and getting values
    simd.set(0, {1, 2, 3});
    simd.set(1, {4, 5, 6});
    simd.set(2, {7, 8, 9});

    assert((simd.get(0) == std::array<int, 3>{1, 2, 3}));
    assert((simd.get(1) == std::array<int, 3>{4, 5, 6}));
    assert((simd.get(2) == std::array<int, 3>{7, 8, 9}));

    // Test sum
    assert((simd.sum() == std::array<int, 3>{12, 15, 18}));

    // Test multiplication
    assert((simd.multiply({1, 1, 1}) == std::array<int, 3>{12, 15, 18}));
    assert((simd.multiply({2, 2, 2}) == std::array<int, 3>{24, 30, 36}));

    // Test clear
    simd.clear();
    assert((simd.get(0) == std::array<int, 3>{0, 0, 0}));
    assert((simd.get(1) == std::array<int, 3>{0, 0, 0}));
    assert((simd.get(2) == std::array<int, 3>{0, 0, 0}));

    // Test out of bounds
    bool caught_exception = false;
    try {
        simd.get(3); // Out of bounds
    } catch (const std::out_of_range&) {
        caught_exception = true;
    }
    assert(caught_exception);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    run_tests();
    return 0;
}
"""

# A trivial fix response, just re-sending the same file in case of errors:
DEBUG_FIX_CODE = DEBUG_VALID_CODE

# ------------------ Setup OpenAI ------------------ #

load_dotenv()  # so OPENAI_API_KEY is available
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------ Utility: Prompt Length Checker ------------------ #

def ensure_prompt_length_ok(prompt, max_length=MAX_PROMPT_LENGTH):
    """
    Raise an error if the prompt is too large.
    """
    if len(prompt) > max_length:
        raise ValueError(f"Prompt length {len(prompt)} exceeded maximum of {max_length} characters. Exiting...")

# ------------------ OpenAI Helpers ------------------ #

def call_openai(prompt, model=MODEL_NAME, temperature=0.2):
    """
    Calls the OpenAI client using the given prompt, unless DEBUG_MODE is True.
    Returns the response text.
    """
    if DEBUG_MODE:
        # Skip calling the API, return a hard-coded successful snippet
        # or a fix snippet, depending on how you want to debug.
        print("[call_openai] DEBUG_MODE is True, returning hard-coded response.\n")
        if "Please fix" in prompt:
            return DEBUG_FIX_CODE
        else:
            return DEBUG_VALID_CODE
    else:
        if PRINT_SEND:
            print("[call_openai] Sending prompt to OpenAI:")
            print("----------------------------------------")
            print(prompt)
            print("----------------------------------------\n")

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
            # temperature=temperature
        )

        answer = response.choices[0].message.content

        print("[call_openai] Received response from OpenAI:")
        print("--------------------------------------------")
        print(answer)
        print("--------------------------------------------\n")

        return answer

# ------------------ File Helpers ------------------ #

def write_to_file(filename, content):
    """
    Overwrites the file with the given content.
    """
    # Ensure the folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"[write_to_file] Writing to {filename}...\n")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

def append_to_file(filename, content):
    """
    Appends the given content to the file (with a separator line).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n")

def read_file(filename):
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
    """
    print(f"[run_executable] Running ./{executable}...\n")
    cmd = [f"./{executable}"]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        return (process.returncode == 0, out.decode("utf-8") + "\n" + err.decode("utf-8"))
    except FileNotFoundError:
        return (False, f"Executable '{executable}' not found. Ensure it was created successfully.")

# ------------------ Main Logic ------------------ #

def main():
    print("[main] Starting code generation & testing process.\n")
    iteration = 0

    # Prompt the user for the problem they want solved
    user_problem = ""
    if not DEBUG_MODE:
        if DEBUG_PROMPT != "":
            user_problem = DEBUG_PROMPT
        else:
            user_problem = input("Enter the problem you want solved: ")

    # Construct the prompt dynamically
    COMBINED_PROMPT = f"""
{GENERATE_PROMPT}
\"\"\"{user_problem}\"\"\"
"""

    print("[main] Requesting single-file C++ code (including tests) from OpenAI...\n")
    ensure_prompt_length_ok(COMBINED_PROMPT)  # Check prompt size
    single_file_code = call_openai(COMBINED_PROMPT)

    # We will iterate up to MAX_ITERATIONS times to fix errors
    while iteration < MAX_ITERATIONS:
        iteration += 1

        # 1) Write the generated code to a new file: generated_v{iteration}.cpp
        cpp_file = os.path.join(GENERATED_FOLDER, f"generated_v{iteration}.cpp")
        write_to_file(cpp_file, single_file_code)

        # Also append the code to everything.txt
        append_to_file(EVERYTHING_FILE, f"===== Iteration {iteration}: Generated Code =====\n{single_file_code}")

        # 2) Compile
        print(f"[main] --- Iteration {iteration} ---\n")
        print("[main] Compiling single-file program...\n")
        success, error_message = compile_cpp(cpp_file, os.path.join(GENERATED_FOLDER, EXECUTABLE))
        if not success:
            # Append compile errors to everything file
            compile_log = f"Compilation Error (Iteration {iteration}):\n{error_message}"
            append_to_file(EVERYTHING_FILE, compile_log)
            print(f"[main] Error compiling:\n{error_message}\n")

            # 3) Request fix
            print("[main] Requesting fix from OpenAI for entire single-file code...\n")
            fix_input = (
                f"{FIX_PROMPT}\n"
                f"Compilation/Verbose Logging Output:\n{error_message}\n"
                f"Current Code:\n{single_file_code}"
            )
            ensure_prompt_length_ok(fix_input)
            single_file_code = call_openai(fix_input)
            continue

        # 4) Run the program (tests)
        print("[main] Running the program (which should include tests)...\n")
        test_success, test_output = run_executable(os.path.join(GENERATED_FOLDER, EXECUTABLE))
        print("[main] Program/Test output:")
        print("--------------------------")
        print(test_output)
        print("--------------------------\n")

        # Save the test output to a file: generated_v{iteration}_output.txt
        output_file = os.path.join(GENERATED_FOLDER, f"generated_v{iteration}_output.txt")
        write_to_file(output_file, test_output)

        # Also append the test output to everything.txt
        append_to_file(EVERYTHING_FILE, f"===== Iteration {iteration}: Test Output =====\n{test_output}")

        if test_success:
            print("[main] The program/test run succeeded (exit code = 0). Tests passed!\n")
            break
        else:
            print("[main] The program/test run failed (non-zero exit). Requesting fix...\n")
            fix_input = (
                f"{FIX_PROMPT}\n"
                f"Runtime/Verbose Logging Output:\n{test_output}\n"
                f"Current Code:\n{single_file_code}"
            )
            ensure_prompt_length_ok(fix_input)
            single_file_code = call_openai(fix_input)

    else:
        print(f"[main] Reached the maximum number of iterations ({MAX_ITERATIONS}) without passing tests.\n")

    print("[main] Process finished.\n")
    print(f"[main] All generated code and outputs have been saved in the '{GENERATED_FOLDER}' folder.")
    print(f"[main] A combined log of everything can be found in '{EVERYTHING_FILE}'.")

if __name__ == "__main__":
    main()
