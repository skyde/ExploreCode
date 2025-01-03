import os
from openai import OpenAI
from dotenv import load_dotenv

class AIService:
    def __init__(self):
        load_dotenv()  # so OPENAI_API_KEY is available
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.print_send = True

    def call_ai(self, system_prompt, user_prompt, model, temperature=0.2):
        """
        Calls the AI service with a system prompt and user prompt, returning the response text.
        """
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
            messages=[{"role": system_role, "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model
        )

        answer = response.choices[0].message.content

        print("[call_ai] Received response from AI service:")
        print("--------------------------------------------")
        print(answer)
        print("--------------------------------------------\n")

        return answer 