"""
python -m models.gpt_runner
"""
import os 
import time
from rich import print
from openai import AzureOpenAI, OpenAI
from openai.types.chat.chat_completion import CompletionUsage


class GPTRunner:
    def __init__(
        self,
        model_name: str,
        run_dir: str = None,
        use_azure: str = "azure",
        ):
        self.model_name = model_name
        if use_azure == "none":
            self.client = OpenAI()
        else:
            self.client = AzureOpenAI(
                    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version = os.getenv("OPENAI_API_VERSION"),
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                )

        self.args_dict = {
            "max_tokens": 8000,
            "temperature": 0.0 
        }
        self.run_dir = run_dir
        
    def single_run_cost(self, usage: CompletionUsage):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        prompt_cost = prompt_tokens * 3 / 1_000_000
        completion_cost = completion_tokens * 15 / 1_000_000
        usage_report = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": prompt_cost + completion_cost,
        }
        return usage_report
        
    def run(self, prompt: str, max_retries: int = 20, retry_delay: float = 20.0):
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
                            **self.args_dict
                        )
                output = response.choices[0].message.content
                usage_report = self.single_run_cost(response.usage)
                return output, usage_report
            
            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print(f"All retry attempts failed. Exception: {e}")
                    return None, None # Make sure don't stop the run

 
if __name__ == "__main__":
    prompt = "What is the sum of 5 and 3?"
    gpt_runner = GPTRunner(model_name="gpt-4o-2")
    response = gpt_runner.run(prompt)
    print(response)