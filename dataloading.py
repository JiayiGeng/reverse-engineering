from tqdm import tqdm
from typing import Any, Dict, List
from dataclasses import asdict, dataclass
from rich import print
from utils import load_data_from_yaml
import random

BLACKBOXES = [
    "programs",
    "languages",
    "ces",
]


@dataclass
class Datapoint:
    sample_id: str = None
    blackbox_name: str = None
    desc_prompt_template: str = None
    func_exp_prompt_template: str = None
    func_imp_prompt_template: str = None
    intv_prompt_template: str = None
    judge_prompt_template: str = None
    reason_prompt_template: str = None
    meta_info: Dict = None
    seed: int = None
    def to_dict(self):
        return asdict(self)


def get_blackbox_datapoints(blackbox_name, sample_id, datapoint, **kwargs):
    prompt_template_path = kwargs.get('prompt_template_path', None)
    assert prompt_template_path is not None, "prompt_template_path is not provided"
    prompt_templates = load_data_from_yaml(prompt_template_path)
    desc_prompt_template = prompt_templates["observation_descriptive"]
    func_exp_prompt_template = prompt_templates['functional_explicit_eval']
    func_imp_prompt_template = prompt_templates['functional_implicit_eval']
    intv_prompt_template = prompt_templates['intervention_prompt'] 
    judge_prompt_template = prompt_templates['judge_prompt']
    reason_prompt_template = prompt_templates['reasoning_intervention_prompt']
    seed = kwargs.get('seed', None)
    if  blackbox_name == "programs":
        meta_info = {
            "program": datapoint['program'],
            "sample_input_output": datapoint['input_output_pairs'],
        }
        
    elif blackbox_name == "languages":
        meta_info = {
            "language_rule": datapoint['language_rule'],
            "language_description": datapoint['description'],
        }
        
    elif blackbox_name == "ces":
        meta_info = {
            "alphas": datapoint['alphas'],
            "names": datapoint['names'],
            "quantity_of_goods": datapoint['quantity_of_goods'],
            "utility_function": datapoint['utility_function'],
            "phi": datapoint['rho'],
        }
    
    datapoint = Datapoint(
        sample_id=sample_id,
        blackbox_name=blackbox_name,            
        desc_prompt_template=desc_prompt_template,
        func_exp_prompt_template=func_exp_prompt_template,
        func_imp_prompt_template=func_imp_prompt_template,
        intv_prompt_template=intv_prompt_template,
        judge_prompt_template=judge_prompt_template,
        reason_prompt_template=reason_prompt_template,
        meta_info=meta_info,
        seed=seed,
    )
    return datapoint


def load_blackbox_dataset(blackbox_name, **kwargs):
    random.seed(0)
    nblackbox_instance = kwargs.get('nblackbox_instance', 10)
    blackbox_config_path = kwargs.get('blackbox_config_path', None)
    difficulty = kwargs.get('difficulty', "none")
    datasets = load_data_from_yaml(blackbox_config_path)
    datapoints = []
    for (sample_id, datapoint) in tqdm(datasets.items(), total=len(datasets), desc="Loading blackbox datapoints"):
        datapoint = get_blackbox_datapoints(blackbox_name, sample_id, datapoint, **kwargs)
        if difficulty != "none" and blackbox_name == "ces":
            if int(datapoint.meta_info['quantity_of_goods']) != int(difficulty):
                continue
        datapoints.append(datapoint)
    datapoints = datapoints[:nblackbox_instance]
    return datapoints


FUNC_TO_BLACKBOX_NAME = {
    load_blackbox_dataset: BLACKBOXES,
}


class DatasetMapping:
    def __init__(self):
        self.blackbox_name_to_func = {
            blackbox_name: func
            for func, blackbox_names in FUNC_TO_BLACKBOX_NAME.items()
            for blackbox_name in blackbox_names
        }
        
    def __call__(self, blackbox_name, **kwargs):
        return self.blackbox_name_to_func[blackbox_name](blackbox_name, **kwargs)