import os
import json
import time
import math
import fire
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from rich import print

from utils import (
    get_available_gpus,
    load_module,
    setup_model,
)
from models.gpt_runner import GPTRunner
from models.vllm_runner import VLLMRunner
from models.claude_runner import ClaudeRunner
from models.dp_runner import DeepSeekRunner
# from configs.configs import SAVE_DIR
from blackboxes.languages import Languages
from blackboxes.programs import Programs
from blackboxes.ces import CES
from evaluation import eval_desc

BLACKBOX_NAME_TO_CLASS = {
    "languages": Languages,
    "ces": CES,
    "programs": Programs,
}

SAVE_DIR = "results"

class ExpLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.blackbox_instance_name = '{blackbox_instance_name}'
        self.save_obs_path = os.path.join(self.save_dir, self.blackbox_instance_name, 'obs.json')
        self.save_desc_results_path = os.path.join(self.save_dir, self.blackbox_instance_name, 'desc_results.json')
        self.save_func_results_path = os.path.join(self.save_dir, self.blackbox_instance_name, 'func_results.json')
        self.save_desc_responses_path = os.path.join(self.save_dir, self.blackbox_instance_name, 'desc_responses.json')
        self.save_func_responses_path = os.path.join(self.save_dir, self.blackbox_instance_name, 'func_responses.json')

    def log_obs(self, obs_datapoints):
        with open(self.save_obs_path, 'w') as f:
            json.dump(obs_datapoints, f, indent=4)

    def log_responses(self, mode, blackbox_instance_name, responses):
        if mode == 'desc':
            save_path = self.save_desc_responses_path.format(blackbox_instance_name=blackbox_instance_name)
        elif mode == 'func':
            save_path = self.save_func_responses_path.format(blackbox_instance_name=blackbox_instance_name)
        with open(save_path, 'w') as f:
            json.dump(responses, f, indent=4)
                
    def log_results(self, mode, blackbox_instance_name, results):
        if mode == 'desc':
            save_path = self.save_desc_results_path.format(blackbox_instance_name=blackbox_instance_name)
        elif mode == 'func':
            save_path = self.save_func_results_path.format(blackbox_instance_name=blackbox_instance_name)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'Results saved | blackbox_instance_name={blackbox_instance_name} | save_path={save_path}')


def main(
        blackbox_name: str, 
        model_name: str,
        model_short_name: str,
        max_model_len: int,
        judge_model_name: str,
        experiment_stages: str,
        run_modes: str,
        nblackbox_instance: int,
        nobs: int,
        nintv: int,
        seed: int, 
        difficulty: str, use_azure: str
    ):
    print(f'Starting experiment at {datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")}')
    num_gpus = get_available_gpus()
    
    # Load datasets
    dataset_module = load_module('dataloading', 'DatasetMapping')
    datapoints = dataset_module()(
        blackbox_name, 
        nblackbox_instance=nblackbox_instance,
        blackbox_config_path=f'configs/{blackbox_name}.yaml',
        prompt_template_path=f'prompts/{blackbox_name}.yaml',
        seed=seed,
        difficulty=difficulty,
    )
    print(f'Loaded {len(datapoints)} datapoints')
    
    run_name = f'm={model_short_name}_exp={experiment_stages}_ndata={len(datapoints)}_nobs={nobs}_nintv={nintv}_dfcty={difficulty}_seed={seed}'
    save_dir = os.path.join(SAVE_DIR, blackbox_name, run_name)
    os.makedirs(save_dir, exist_ok=True)
    experiment_stages = experiment_stages.split('+')
    run_modes = run_modes.split('+')
    exp_logger = ExpLogger(save_dir)
    model_runner = None
    llm_judge = None
    
    print(f'Running modes: {run_modes}')
    print(f'Experiment stages: {experiment_stages}')

    if "datagen" in run_modes:
        obs_module = load_module('observations', 'ObservationMapping')
        intv_module = load_module('interventions', 'InterventionMapping')
        batched_intv_datapoints = []
        if "obs" in experiment_stages:
            for datapoint in tqdm(datapoints, desc='Generating obs datapoints', total=len(datapoints)):
                blackbox_instance_name = datapoint.sample_id
                save_obs_datapoints_path = os.path.join(save_dir, f'{blackbox_instance_name}', f'obs.json')
                if not os.path.exists(os.path.join(save_dir, f'{blackbox_instance_name}')):
                    os.makedirs(os.path.join(save_dir, f'{blackbox_instance_name}'), exist_ok=True)
                if os.path.exists(save_obs_datapoints_path):
                    try:
                        with open(save_obs_datapoints_path, 'r') as f:
                            save_obj = json.load(f)
                            obs_for_eval = save_obj['obs_for_eval']
                            print(f'skipping {blackbox_instance_name} because it already exists')
                            continue
                    except:
                        print(f'Getting observational datapoints for {blackbox_instance_name}')
                
                blackbox = BLACKBOX_NAME_TO_CLASS[blackbox_name](f'configs/{blackbox_name}.yaml', blackbox_instance_name)
                obs_datapoints = obs_module()(
                    blackbox,
                    blackbox_name,
                    datapoint,
                    nobs=nobs,
                )
                obs_for_eval = obs_datapoints

                if "intv" in experiment_stages:
                    # Prepare intv for batch inference
                    blackbox = BLACKBOX_NAME_TO_CLASS[blackbox_name](f'configs/{blackbox_name}.yaml', blackbox_instance_name)
                    intv_datapoint = {
                        "sample_id": blackbox_instance_name,
                        "nobs": nobs,
                        "nintv": nintv,
                        "obs_datapoints": obs_datapoints,
                        "blackbox": blackbox,
                        "blackbox_name": blackbox_name,
                        "datapoint": datapoint,
                    } 
                    batched_intv_datapoints.append(intv_datapoint)
                else:
                    with open(save_obs_datapoints_path, 'w') as f:
                        save_obj = {
                            'obs_datapoints': obs_for_eval,
                            'intv_datapoints': [],
                            'intv_responses': [],
                            'obs_for_eval': obs_for_eval,
                        }
                        json.dump(save_obj, f, indent=4)
                        f.close()
                        print(f'Saved sample_id={blackbox_instance_name} to: {save_obs_datapoints_path}')

        if len(batched_intv_datapoints) > 0 and "intv" in experiment_stages:
            if "reason" in experiment_stages:
                intv_mode = "reason"
            elif "hypodesc" in experiment_stages:
                intv_mode = "hypodesc"
            elif "hypofunc" in experiment_stages:
                intv_mode = "hypofunc"
            else:
                intv_mode = "intv"
            if "gpt" in model_short_name:
                gpt_runner = GPTRunner(model_name=model_name, use_azure=use_azure)
                model_runner = gpt_runner
                intv_lm_responses = []
                intv_datapoints = []
                for intv_datapoint in tqdm(batched_intv_datapoints, desc='Generating intv datapoints', total=len(batched_intv_datapoints)):
                    intv_lm_response, intv_datapoint = intv_module()(
                        blackbox_name,
                        [intv_datapoint],
                        model_runner,
                        intv_mode=intv_mode, 
                    )
                    intv_lm_responses.extend(intv_lm_response)
                    intv_datapoints.extend(intv_datapoint)
            elif "claude" in model_short_name:
                claude_runner = ClaudeRunner(model_name=model_name)
                model_runner = claude_runner
                intv_lm_responses = []
                intv_datapoints = []
                for intv_datapoint in tqdm(batched_intv_datapoints, desc='Generating intv datapoints', total=len(batched_intv_datapoints)):
                    intv_lm_response, intv_datapoint = intv_module()(
                        blackbox_name,
                        [intv_datapoint],
                        model_runner,
                        intv_mode=intv_mode, 
                    )
                    intv_lm_responses.extend(intv_lm_response)
                    intv_datapoints.extend(intv_datapoint)
            elif "deepseek" in model_short_name:
                deepseek_runner = DeepSeekRunner(model_name=model_name)
                model_runner = deepseek_runner
                intv_lm_responses = []
                intv_datapoints = []
                for intv_datapoint in tqdm(batched_intv_datapoints, desc='Generating intv datapoints', total=len(batched_intv_datapoints)):
                    intv_lm_response, intv_datapoint = intv_module()(
                        blackbox_name,
                        [intv_datapoint],
                        model_runner,
                        intv_mode=intv_mode, 
                    )
                    intv_lm_responses.extend(intv_lm_response)
                    intv_datapoints.extend(intv_datapoint)
                    
            else:
                model, tokenizer = setup_model(
                    model_name=model_name,
                    max_model_len=max_model_len,
                    trust_remote_code=True,
                    num_gpus=num_gpus,
                )
                model_runner = VLLMRunner(model=model, tokenizer=tokenizer)

                intv_lm_responses, intv_datapoints = intv_module()(
                    blackbox_name,
                    batched_intv_datapoints,
                    model_runner,
                    intv_mode=intv_mode,
                )
            for intv_response, intv_datapoint, datapoint in zip(intv_lm_responses, intv_datapoints, datapoints):
                blackbox_instance_name = datapoint.sample_id
                save_obs_datapoints_path = os.path.join(save_dir, f'{blackbox_instance_name}', f'obs.json')
                with open(save_obs_datapoints_path, 'w') as f:
                    save_obj = {
                        'obs_datapoints': obs_for_eval,
                        'intv_datapoints': intv_datapoint,
                        'intv_responses': intv_response,
                        'obs_for_eval': intv_datapoint
                    }
                    json.dump(save_obj, f, indent=4)
                f.close()
                print(f'Saved sample_id={blackbox_instance_name} to: {save_obs_datapoints_path}')

    if "eval" in run_modes:
        print("Evaluating...")
        if "gpt" not in model_short_name and "claude" not in model_short_name and "deepseek" not in model_short_name:
            model, tokenizer = setup_model(
                model_name=model_name,
                max_model_len=max_model_len,
                trust_remote_code=True,
                num_gpus=num_gpus,
            )
            model_runner = VLLMRunner(model=model, tokenizer=tokenizer)
        desc_scores = []
        batched_for_eval = []
        for datapoint in tqdm(datapoints, desc='Running evaluation...', total=len(datapoints)):
            if "gpt" in model_short_name:
                gpt_runner = GPTRunner(model_name=model_name, use_azure=use_azure)
                model_runner = gpt_runner
            elif "claude" in model_short_name:
                claude_runner = ClaudeRunner(model_name=model_name)
                model_runner = claude_runner
            elif "deepseek" in model_short_name:
                deepseek_runner = DeepSeekRunner(model_name=model_name)
                model_runner = deepseek_runner
            blackbox_instance_name = datapoint.sample_id
            desc_results_path = exp_logger.save_desc_results_path.format(blackbox_instance_name=blackbox_instance_name)
            save_obs_path = exp_logger.save_obs_path.format(blackbox_instance_name=blackbox_instance_name)
            if os.path.exists(desc_results_path):
                try:
                    with open(desc_results_path, 'r') as f:
                        desc_results = json.load(f)
                        desc_scores.append(desc_results['score'])
                        print(f'Skipping {blackbox_instance_name} because it already exists')
                        continue
                except:
                    print(f'Running evaluation for {blackbox_instance_name}')
                    
            try:
                with open(save_obs_path, 'r') as f:
                    save_obj = json.load(f)
                obs_for_eval = save_obj['obs_for_eval']
            except:
                raise ValueError(f'No observational datapoints for blackbox_instance_name={blackbox_instance_name}')
            
            batched_for_eval.append((
                datapoint,
                obs_for_eval,
            ))
        
        # split batch into chunks of batch_size
        batch_size = 4 if "gpt" not in model_short_name else 1
        num_batches = math.ceil(len(batched_for_eval) / batch_size)
        for i in range(num_batches):
            batch_chunk = batched_for_eval[i * batch_size:(i + 1) * batch_size]
            desc_responses, metrics = eval_desc(
                blackbox_name,
                batch_chunk,
                model_runner,
            )
            for desc_response, (datapoint, _) in zip(desc_responses, batch_chunk):
                blackbox_instance_name = datapoint.sample_id
                exp_logger.log_responses('desc', blackbox_instance_name, desc_response)
                print(f'Saved {blackbox_instance_name} to {exp_logger.save_desc_responses_path.format(blackbox_instance_name=blackbox_instance_name)}')
        
    if "judge" in run_modes:
        print("Judging...")
        assert 'gpt' in judge_model_name
        llm_judge = GPTRunner(model_name=judge_model_name, use_azure=use_azure)
        desc_scores = []
        for datapoint in tqdm(datapoints, desc='Judging...', total=len(datapoints)):
            blackbox_instance_name = datapoint.sample_id
            desc_results_path = exp_logger.save_desc_results_path.format(blackbox_instance_name=blackbox_instance_name)
            desc_responses_path = exp_logger.save_desc_responses_path.format(blackbox_instance_name=blackbox_instance_name)
            with open(desc_responses_path, 'r') as f:
                desc_responses = json.load(f)
            judge_reponses, judge_metrics = eval_desc(
                blackbox_name = blackbox_name,
                batched_for_eval = [(datapoint, desc_responses)],
                model_runner = None,
                llm_judge = llm_judge,
                desc_prompt = desc_responses['desc_prompt'],
                desc_response = desc_responses['desc_response'],
                desc_usage_reports = desc_responses['desc_usage_reports'],
            )
            desc_scores.append(judge_metrics['score'])
            with open(desc_responses_path, 'w') as f:
                json.dump(judge_reponses, f, indent=4)
            print(f'Descriptive responses saved to: {desc_responses_path}')

            desc_results_path = Path(desc_responses_path).parent / 'desc_results.json'
            with open(desc_results_path, 'w') as f:
                json.dump(judge_metrics, f, indent=4)
            print(f'Descriptive results saved to: {desc_results_path}')
            
        overall_results_path = os.path.join(save_dir, 'metrics.json')
        if os.path.exists(overall_results_path):
            with open(overall_results_path, 'r') as f:
                overall_results = json.load(f)
                overall_results['desc_score'] = np.mean(desc_scores)
            with open(overall_results_path, 'w') as f:
                json.dump(overall_results, f, indent=4)
            print(f'Overall results saved to: {overall_results_path}')
        else:
            overall_results = dict(
                desc_score=np.mean(desc_scores),
            )
            with open(overall_results_path, 'w') as f:
                json.dump(overall_results, f, indent=4)
            print(f'Overall results saved to: {overall_results_path}')
            
        print(f'Overall results: {overall_results}')
    

if __name__ == "__main__":
    fire.Fire(main)