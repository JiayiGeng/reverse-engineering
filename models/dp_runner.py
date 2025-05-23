"""
python -m models.dp_runner
"""
import os
import time
from rich import print
from together import Together


class DeepSeekRunner:
    def __init__(
        self,
        model_name: str,
        run_dir: str = None
        ):
        self.model_name = model_name

        self.client = Together()
        self.run_dir = run_dir

    def run(self, prompt: str, max_retries: int = 5, retry_delay: float = 5.0):
        if isinstance(prompt, list):
            messages = [
                {
                    "role": "user",
                    "content": prompt[0]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=128000,
                        )

                output = response.choices[0].message.content
                return output, "none"

            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print(f"All retry attempts failed. Exception: {e}")
                    return None, None # Make sure don't stop the run


if __name__ == "__main__":
    prompt = "What is the sum of 5 and 3?"
    gpt_runner = DeepSeekRunner(model_name="deepseek-ai/DeepSeek-R1")
    response = gpt_runner.run(prompt)