from rich import print
import random
import copy

def get_languages(blackbox, blackbox_name, datapoint, **kwargs):
    nobs = kwargs.get('nobs', 0)
    obs_datapoints = ""
    for _ in range(nobs):
        input_string = blackbox.run_language(mode="observation")
        obs_datapoints += input_string + "\n" 
    return obs_datapoints[:-len("\n")]

def get_programs(blackbox, blackbox_name, datapoint, **kwargs):
    nobs = kwargs.get('nobs', 0)
    obs_datapoints = ""
    input_lists = []
    for _ in range(nobs):
        input_len = random.randint(1, 10)
        input_list = [random.randint(0, 100) for _ in range(input_len)]
        input_lists.append(input_list)
    for input_list in input_lists:
        output_list = blackbox.run_program(input_list = input_list, mode = "observation")
        obs_datapoints += f"Input: {input_list}\nOutput: {output_list}\n"
    return obs_datapoints[:-len("\n")]

def get_ces(blackbox, blackbox_name, datapoint, **kwargs):
    nobs = kwargs.get('nobs', 0)
    obs_datapoints = ""
    for _ in range(nobs):
        preference, basket_quantity1, basket_quantity2 = blackbox.run_ces(mode="observation")
        obs_datapoints += f"Basket1: {basket_quantity1}\nBasket2: {basket_quantity2}\n{preference}" + "\n"
    return obs_datapoints[:-len("\n")]


OBS_TO_BLACKBOX_NAME = {
    get_languages: ["languages"],
    get_programs: ["programs"],
    get_ces: ["ces"],
}


class ObservationMapping:
    def __init__(self):
        self.blackbox_name_to_obs = {
            blackbox_name: obs
            for obs, blackbox_names in OBS_TO_BLACKBOX_NAME.items()
            for blackbox_name in blackbox_names
        }
        
    def __call__(self, blackbox, blackbox_name, datapoint, **kwargs):
        return self.blackbox_name_to_obs[blackbox_name](blackbox, blackbox_name, datapoint, **kwargs)
