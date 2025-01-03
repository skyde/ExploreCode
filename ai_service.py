import os
from openai import OpenAI
from dotenv import load_dotenv

class AIService:
    def __init__(self, debug_mode=False):
        """
        :param debug_mode: If True, call_ai will return a series of test C++ code snippets and skip actual API calls.
        """
        load_dotenv()  # so OPENAI_API_KEY is available
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.print_send = True
        self.debug_mode = debug_mode
        
        # This counter tracks the number of times we've generated debug code
        self.debug_calls = 0

    def call_ai(self, system_prompt, user_prompt, model, temperature=0.2):
        """
        Calls the AI service with a system prompt and user prompt, returning the response text.
        If debug_mode is True, returns progressively changing C++ code snippets on each call.
        """
        # If debug mode is enabled, return progressively changing C++ sample code.
        if self.debug_mode:
            self.debug_calls += 1
            
            if self.debug_calls == 1:
                # Returns code that doesn't compile
                sample_code = (
                    '#include <iostream>\n\n'
                    'int main() {\n'
                    '    // This code does not compile\n'
                    '    std::cout << "First test: does not compile" << std::endl;\n'
                    '    // Oops: missing semicolon or some random syntax error\n'
                    '    // Let\'s force a compile error by referencing an undefined symbol\n'
                    '    undefined_symbol();\n'
                    '}\n'
                )
            elif self.debug_calls == 2:
                # Returns code that compiles but exits with a non-zero code and prints "test failed"
                sample_code = (
                    '#include <iostream>\n\n'
                    'int main() {\n'
                    '    std::cout << "test failed" << std::endl;\n'
                    '    return 1; // Non-zero exit code\n'
                    '}\n'
                )
            else:
                # Third and subsequent times: returns a working hello world
                sample_code = (
                    '#include <iostream>\n\n'
                    'int main() {\n'
                    '    std::cout << "Hello World" << std::endl;\n'
                    '    std::cout << "All tests passed" << std::endl;\n'
                    '    return 0;\n'
                    '}\n'
                )

            if self.print_send:
                print(f"[DEBUG MODE] Debug call #{self.debug_calls}, returning sample C++ code.")
            return sample_code

        # Normal (non-debug) behavior:
        if self.print_send:
            print("[call_ai] Sending prompt to AI service:")
            print("------------------ System --------------")
            print(system_prompt)
            print("------------------ User ----------------\n")
            print(user_prompt)
            print("----------------------------------------\n")

        # For non-supported models, just use the user role
        system_role = "user"

        response = self.client.chat.completions.create(
            messages=[
                {"role": system_role, "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
            model=model
        )

        answer = response.choices[0].message.content

        print("[call_ai] Received response from AI service:")
        print("--------------------------------------------")
        print(answer)
        print("--------------------------------------------\n")

        return answer
