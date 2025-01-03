import subprocess

class CppCompiler:
    """
    A small helper class to compile and run C++ code with clang++.
    """

    def compile_cpp(self, source_file, output_file):
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

    def run_executable(self, executable):
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
