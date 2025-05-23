import yaml
import json
import importlib
from transformers import AutoTokenizer
from vllm import LLM
import math
import re
import ast
import torch
from collections import Counter
import ast
import numpy as np


def load_data_from_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def load_module(module_path, mapping):
    module = importlib.import_module(module_path)
    return getattr(module, mapping)


def get_available_gpus():
    return torch.cuda.device_count()


def setup_model(model_name, max_model_len, trust_remote_code, num_gpus=4):
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=num_gpus,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer


def extract_answer(text):
    pattern = r"```answer\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    answer_text = match.group(1).strip()
    answer = {}
    for line in answer_text.splitlines():
        line = line.strip()
        if not line:
            continue
       
        if ":" not in line:
            continue
        key, value_str = line.split(":", 1)
        key = key.strip()
        value_str = value_str.strip()
      
        if value_str.startswith("[") and value_str.endswith("]"):
            try:
                value = ast.literal_eval(value_str)
            except Exception:
                value = value_str
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
        answer[key] = value
    return answer


def compute_rmse(answer: dict, ground_truth: dict) -> dict:
    """
    Compute RMSE for alpha parameters and rho between answer and ground truth,
    ensuring individual errors are clamped to [0,1], total_rmse ≤ 1,
    and preserving flip_rmse = 1 - total_rmse.
    """
    # 1) Compute alpha RMSE
    a_i_pred = np.array(answer.get('a_i', []), dtype=float)
    a_i_true = np.array(ground_truth.get('a_i', []), dtype=float)
    
    if a_i_pred.size and a_i_true.size and a_i_pred.shape == a_i_true.shape:
        a_i_rmse = math.sqrt(np.mean((a_i_pred - a_i_true) ** 2))
    else:
        a_i_rmse = float('nan')
    
    # 2) Compute rho RMSE
    try:
        rho_pred = float(answer.get('rho', np.nan))
        rho_true = float(ground_truth.get('rho', np.nan))
        rho_rmse = abs(rho_pred - rho_true)
    except (TypeError, ValueError):
        rho_rmse = float('nan')
    
    # 3) Clamp individual errors to [0,1]
    ai_clamped = 0.0 if math.isnan(a_i_rmse) else max(0.0, min(1.0, a_i_rmse))
    pr_clamped = 0.0 if math.isnan(rho_rmse) else max(0.0, min(1.0, rho_rmse))
    
    # 4) Combined RMSE normalized to [0,1]
    total_rmse = math.sqrt(ai_clamped**2 + pr_clamped**2) / math.sqrt(2)
    
    # 5) Flip RMSE
    flip_rmse = 1.0 - total_rmse
    
    return {
        'a_i_rmse': a_i_rmse,
        'rho_rmse': rho_rmse,
        'total_rmse': total_rmse,
        'flip_rmse': flip_rmse
    }
    

def extract_score(text):
    pattern = r'```score\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return ""
    

def extract_judgement(text):
    pattern = r'```judgement\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text
    

def create_judge_prompt(judge_prompt_template, ground_truth, response):
    judge_prompt = judge_prompt_template.format(
        ground_truth=ground_truth,
        response=response,
    )
    return judge_prompt


def create_prompt(prompt_template, observation_data = "none", in_outs = "none"):
    if observation_data != "none" and in_outs != "none":
        prompt = prompt_template.format(
            observations=observation_data,
            input_examples=in_outs[0],
            output_examples=in_outs[1]
            )
    elif observation_data != "none":
        prompt = prompt_template.format(
            observations=observation_data,
            )
    return prompt


def extract_rules(text):
    try:
        pattern = r'```rules\s*\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text
    except Exception as e:
        print(text)
        input()
        print(f"Error extracting rules: {e}")
        return text
    

def extract_query(query_text, blackbox_name):
    if blackbox_name == "programs":
        return extract_list(query_text)
    elif blackbox_name == "languages":
        return extract_language(query_text)
    else:
        return "none"
    

def extract_list(text: str):
    lines = text.splitlines()
    in_query_block = False
    query_content_lines = []

    # Collect lines inside the query block
    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("```query") or stripped_line.startswith("```test"):
            in_query_block = True
            continue

        if in_query_block and stripped_line.startswith("```"):
            in_query_block = False
            break

        if in_query_block:
            # Keep non-empty lines (you can decide whether to keep empty lines if needed)
            if stripped_line:
                query_content_lines.append(stripped_line)

    # Parse each line as a Python list
    parsed_lists = []
    for content_line in query_content_lines:
        try:
            result = ast.literal_eval(content_line)
            if isinstance(result, list):
                parsed_lists.append(result)
            else:
                parsed_lists.append(None)
        except Exception:
            parsed_lists.append(None)

    # Prepare to return exactly two lists
    first_list = parsed_lists[0] if len(parsed_lists) > 0 else "none"
    second_list = parsed_lists[1] if len(parsed_lists) > 1 else "none"

    return first_list, second_list


def extract_language(text):
    pattern = r'```test\s*\n?(.*?)\n?```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    pattern = r'```test\s*\n?(.*?)\n?'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return text[len("```test\n"):]
    
    return text


def extract_baskets(text: str):
    pattern = r'\[.*?\]'
    matches = re.findall(pattern, text)
    baskets = []
    for m in matches:
        try:
            val = ast.literal_eval(m)
            baskets.append(val)
        except Exception as e:
            continue
    if len(baskets) < 2:
        raise ValueError("Not enough valid basket data found in the text.")
    return baskets[0], baskets[1]


def extract_pred_ces(text):
    # 1. Extract the content of the Func code block
    match = re.search(r'```Func(.*?)```', text, re.S)
    if not match:
        raise ValueError("No ```Func``` code block found.")
    func_block = match.group(1)

    # 2. Find all a_i coefficients (e.g., "0.6 \\cdot x_1")
    ai_matches = re.findall(r'([0-9]*\.?[0-9]+)\s*\\cdot\s*x_\d+', func_block)
    a_i = [float(val) for val in ai_matches]
    if not a_i:
        raise ValueError("No coefficients a_i found in the Func block.")

    # 3. Extract rho (exponent on x_i inside the sum)
    rho_matches = re.findall(r'x_\d+\^\{?([0-9]*\.?[0-9]+)\}?', func_block)
    if not rho_matches:
        rho = 0.0
    else:
        rho = float(rho_matches[0])

    return {'a_i': a_i, 'rho': rho}

