"""
python -m models.claude_runner
"""
import os 
import time
from rich import print
import anthropic


class ClaudeRunner:
    def __init__(
        self,
        model_name: str,
        run_dir: str = None
        ):
        self.model_name = model_name
       
        self.client = anthropic.Anthropic()

        self.args_dict = {
            "max_tokens": 64000 if "3-7" in model_name else 8000,
            "temperature": 0.0,
        }
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
                response = self.client.messages.create(
                            model=self.model_name,
                            messages=messages,
                            **self.args_dict
                        )
                output = response.content[0].text
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
    gpt_runner = ClaudeRunner(model_name="claude-3-5-sonnet-20240620")
    response = gpt_runner.run(prompt)
    print(response)