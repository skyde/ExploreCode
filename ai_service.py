import os
from openai import OpenAI
from dotenv import load_dotenv

class AIService:
    def __init__(self, debug_mode=False):
        """
        :param debug_mode: If True, call_ai will return sample code and skip actual API calls.
        """
        load_dotenv()  # so OPENAI_API_KEY is available
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.print_send = True
        self.debug_mode = debug_mode

    def call_ai(self, system_prompt, user_prompt, model, temperature=0.2):
        """
        Calls the AI service with a system prompt and user prompt, returning the response text.
        If debug_mode is True, returns a sample 'Hello World' code snippet instead of calling the API.
        """
        # If debug mode is enabled, simply return a sample code snippet
        if self.debug_mode:
            sample_code = (
                "# Sample Python code for testing\n"
                "def main():\n"
                "    print('Hello, World!')\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
            if self.print_send:
                print("[DEBUG MODE] Skipping API call and returning sample code.")
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
            model=model,
            temperature=temperature
        )

        answer = response.choices[0].message.content

        print("[call_ai] Received response from AI service:")
        print("--------------------------------------------")
        print(answer)
        print("--------------------------------------------\n")

        return answer
