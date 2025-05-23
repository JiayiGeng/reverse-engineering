import json
import ast
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
from rich import print
from utils import (
    extract_score, 
    extract_judgement, 
    create_judge_prompt, 
    create_prompt,
    extract_rules,
    extract_pred_ces,
    compute_rmse
)
from models.gpt_runner import GPTRunner
from models.claude_runner import ClaudeRunner
from models.dp_runner import DeepSeekRunner


def eval_desc(
    blackbox_name,
    batched_for_eval,
    model_runner = None,
    llm_judge = None,
    **kwargs,
):
    print(f"Evaluating {blackbox_name} with descriptive mode using {model_runner}")
    assert (model_runner is not None and llm_judge is None) or (model_runner is None and llm_judge is not None)
    if model_runner is not None:
        desc_prompts = [
            create_prompt(
                datapoint.desc_prompt_template,
                observation_data=observations,
            )
            for datapoint, observations in batched_for_eval
        ]
        if isinstance(model_runner, GPTRunner) or isinstance(model_runner, ClaudeRunner) or isinstance(model_runner, DeepSeekRunner):
            desc_responses, usage_reports = model_runner.run(desc_prompts[0])
            metrics = None
            responses = [
                dict(
                    desc_prompt=desc_prompt,
                    desc_response=desc_responses,
                    desc_usage_reports=usage_reports,
                    meta_info=datapoint[0].meta_info,
                )
                for datapoint, desc_prompt in zip(batched_for_eval, desc_prompts)
            ]
            return responses, metrics
        else: # VLLM
            desc_responses, usage_reports = model_runner.run(desc_prompts)
            metrics = None
            responses = [
                dict(
                    desc_prompt=desc_prompt,
                    desc_response=desc_response,
                    desc_usage_reports=usage_report,
                    meta_info=datapoint[0].meta_info,
                )
                for datapoint, desc_prompt, desc_response, usage_report in zip(batched_for_eval, desc_prompts, desc_responses, usage_reports)
            ]
            return responses, metrics
    
    elif llm_judge is not None:
        print(f"Using LLM judge: {llm_judge}")
        assert len(batched_for_eval) == 1
        datapoint, desc_response = batched_for_eval[0]
        desc_prompt = kwargs.get('desc_prompt', None)
        desc_response = kwargs.get('desc_response', None)
        desc_usage_reports = kwargs.get('desc_usage_reports', None)
        desc_judge_prompt = create_judge_prompt(
            datapoint.judge_prompt_template,
            ground_truth=datapoint.meta_info,
            response=extract_rules(desc_response),
        )
        desc_judge_response, usage_report = llm_judge.run(desc_judge_prompt)
        try:
            score = extract_score(desc_judge_response)
            judgement = extract_judgement(desc_judge_response)
        except Exception as e:
            print(f"Error extracting score or judgement: {e}")
            score = None
            judgement = None
        if blackbox_name == "ces":
            answer = extract_pred_ces(desc_response)
            ground_truth = {
                "a_i": datapoint.meta_info["alphas"],
                "rho": datapoint.meta_info['phi'],
            }
            rmse = compute_rmse(answer, ground_truth)
        metrics = {
            "score": float(score) * 10,
            "judgement": judgement,
            "rmse": rmse if blackbox_name == "ces" else None,
        }
        responses = {
            "desc_prompt": desc_prompt,
            "desc_response": desc_response,
            "desc_usage_reports": desc_usage_reports,
            "desc_judge_prompt": desc_judge_prompt,
            "desc_judge_response": desc_judge_response,
            "desc_judge_usage_report": usage_report,
            "meta_info": datapoint.meta_info,
        }
        return responses, metrics