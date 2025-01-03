import re

DEBUG_PROMPT = """
- Generate an algorithm that calculates connectivity clustering given a graph of nodes and edges
- Each node is connected to 1-8 other nodes
- The algorithm should be very efficient, and create benchmarks to verify performance
- Use a 'bit matrix' to accelerate the connectivity clustering, using 256 bit registers, and clever use of intrinsics and bit manipulation operations to accelerate (e.g. bitwise OR)
- Use SIMD operations (AVX2)
- Only use a single core
"""

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
- Do NOT put code in
cpp blocks.
"""

def sanitize_for_filename(text, max_length=30):
    """
    Converts text into a safe snippet for folder names by:
      - Replacing all non-alphanumeric or underscore/dash with underscores
      - Truncating to a certain max_length
    """
    # Replace disallowed chars with underscores
    safe_text = re.sub(r'[^a-zA-Z0-9_\-]+', '_', text)
    safe_text = safe_text.strip("_")
    # Truncate
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]
    return safe_text
