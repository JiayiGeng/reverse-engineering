from vllm import SamplingParams


class VLLMRunner:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens=12800,
        temperature=0.0,
        top_p=1.0,
        stop_tokens=[],
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
        )
        
    def run(self, prompts):
        """
        Currently only supports single prompt per inference.
        """
        output_type = "list"
        if isinstance(prompts, str): # For intv
            prompts = [prompts]
            output_type = "text"
            
        input_texts = []
        for prompt in prompts:
            messages = [dict(role='user', content=prompt)]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_texts.append(input_text)
        outputs = self.model.generate(input_texts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        if output_type == "text":
            return generated_texts[0], 'N/A'
        else:
            return generated_texts, ['N/A' for _ in range(len(generated_texts))]

