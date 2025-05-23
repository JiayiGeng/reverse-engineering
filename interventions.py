from rich import print
from tqdm import tqdm
from utils import (
    extract_query,
    extract_baskets,
)


def get_languages(batched_intv_datapoints, agent, **kwargs):
    nintv_left = [intv_datapoint['nintv'] for intv_datapoint in batched_intv_datapoints]
    intv_datapoints = [intv_datapoint['obs_datapoints'] for intv_datapoint in batched_intv_datapoints]
    intv_lm_responses = [[] for _ in range(len(batched_intv_datapoints))]
    intv_mode = kwargs.get('intv_mode', 'intv')
    get_intv = True
    while get_intv:
        intv_prompts = []
        if intv_mode == "intv":
            for batched_intv_datapoint, intv_datapoint in zip(batched_intv_datapoints, intv_datapoints):
                intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                    observations = intv_datapoint
                ))
        elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
            for idx, (batched_intv_datapoint, intv_datapoint) in enumerate(zip(batched_intv_datapoints, intv_datapoints)):
                if nintv_left[idx] % 5 == 0:
                    if intv_mode == "reason":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].reason_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypodesc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_imp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypofunc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_exp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                else:
                    intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                        observations = intv_datapoint
                    ))
        intv_responses, _ = agent.run(intv_prompts)
        if isinstance(intv_responses, str):
            intv_responses = [intv_responses] 
        try:
            for idx, intv_response in enumerate(intv_responses):
                intv_lm_responses[idx].append(intv_response)
        except Exception as e:
            return intv_lm_responses, intv_datapoints
        for idx, intv_response in enumerate(intv_responses):
            if intv_mode == "intv":
                query = extract_query(intv_response, batched_intv_datapoints[idx]['blackbox_name'])
                intv_data = batched_intv_datapoints[idx]['blackbox'].run_language(mode="intervention", query=query)
                intv_datapoints[idx] = intv_datapoints[idx] + "\n" + intv_data
            elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
                if nintv_left[idx] % 5 == 0:
                    intv_datapoints[idx] = intv_datapoints[idx] + "\n" + intv_response
                query = extract_query(intv_response, batched_intv_datapoints[idx]['blackbox_name'])
                intv_data = batched_intv_datapoints[idx]['blackbox'].run_language(mode="intervention", query=query)
                intv_datapoints[idx] = intv_datapoints[idx] + "\n" + intv_data
        nintv_left = [nintv_left - 1 for nintv_left in nintv_left]
        if all(x == 0 for x in nintv_left):
            get_intv = False
    return intv_lm_responses, intv_datapoints
        
        
def get_programs(batched_intv_datapoints, agent, **kwargs):
    nintv_left = [intv_datapoint['nintv'] for intv_datapoint in batched_intv_datapoints]
    intv_datapoints = [intv_datapoint['obs_datapoints'] for intv_datapoint in batched_intv_datapoints]
    intv_lm_responses = [[] for _ in range(len(batched_intv_datapoints))]
    intv_mode = kwargs.get('intv_mode', 'intv')
    get_intv = True
    while get_intv:
        intv_prompts = []
        if intv_mode == "intv":
            for batched_intv_datapoint, intv_datapoint in zip(batched_intv_datapoints, intv_datapoints):
                intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                    observations = intv_datapoint
                ))
        elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
            for idx, (batched_intv_datapoint, intv_datapoint) in enumerate(zip(batched_intv_datapoints, intv_datapoints)):
                if nintv_left[idx] % 5 == 0:
                    if intv_mode == "reason":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].reason_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypodesc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_imp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypofunc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_exp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                else:
                    intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                        observations = intv_datapoint
                    ))
        intv_responses, _ = agent.run(intv_prompts)
        if isinstance(intv_responses, str):
            intv_responses = [intv_responses] 
        for idx, intv_response in enumerate(intv_responses):
            intv_lm_responses[idx].append(intv_response)
        for idx, intv_response in enumerate(intv_responses):
            if intv_mode == "intv":
                input_list, output_list = extract_query(intv_response, batched_intv_datapoints[idx]['blackbox_name'])
                if input_list == "none":
                    intv_datapoint = "Input: \n" + "none" + "\n" + "\nOutput: \n" + "Invalid query" + "\n\n"
                elif output_list == "none":
                    intv_data = batched_intv_datapoints[idx]['blackbox'].run_program(input_list = input_list, mode = "intervention")
                    intv_datapoints[idx] = intv_datapoints[idx] + "Input: \n" + str(input_list) + "\n" + "\nOutput: \n" + str(intv_data) + "\n\n"
                elif input_list != "none" and output_list != "none":
                    intv_data = batched_intv_datapoints[idx]['blackbox'].run_program(input_list = input_list, output_list = output_list, mode = "intervention")
                    intv_datapoints[idx] = intv_datapoints[idx] + "Input: \n" + str(input_list) + "\n" + "\nOutput: \n" + str(intv_data) + "\n\n"
            elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
                if nintv_left[idx] % 5 == 0:
                    intv_datapoints[idx] = intv_datapoints[idx] + "\n" + intv_response
                input_list, output_list = extract_query(intv_response, batched_intv_datapoints[idx]['blackbox_name'])
                if input_list == "none":
                    intv_datapoint = "Input: \n" + "none" + "\n" + "\nOutput: \n" + "Invalid query" + "\n\n"
                elif output_list == "none":
                    intv_data = batched_intv_datapoints[idx]['blackbox'].run_program(input_list = input_list, mode = "intervention")
                    intv_datapoints[idx] = intv_datapoints[idx] + "Input: \n" + str(input_list) + "\n" + "\nOutput: \n" + str(intv_data) + "\n\n"
                else:
                    intv_data = batched_intv_datapoints[idx]['blackbox'].run_program(input_list = input_list, output_list = output_list, mode = "intervention")
                    intv_datapoints[idx] = intv_datapoints[idx] + "Input: \n" + str(input_list) + "\n" + "\nOutput: \n" + str(intv_data) + "\n\n"
                    
        nintv_left = [nintv_left - 1 for nintv_left in nintv_left]
        if all(x == 0 for x in nintv_left):
            get_intv = False
    return intv_lm_responses, intv_datapoints

def get_ces(batched_intv_datapoints, agent, **kwargs):
    nintv_left = [intv_datapoint['nintv'] for intv_datapoint in batched_intv_datapoints]
    intv_datapoints = [intv_datapoint['obs_datapoints'] for intv_datapoint in batched_intv_datapoints]
    intv_lm_responses = [[] for _ in range(len(batched_intv_datapoints))]
    intv_mode = kwargs.get('intv_mode', 'intv')
    get_intv = True
    while get_intv:
        intv_prompts = []
        if intv_mode == "intv":
            for batched_intv_datapoint, intv_datapoint in zip(batched_intv_datapoints, intv_datapoints):
                intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                    observations = intv_datapoint
                ))
        elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
            for idx, (batched_intv_datapoint, intv_datapoint) in enumerate(zip(batched_intv_datapoints, intv_datapoints)):
                if nintv_left[idx] % 5 == 0:
                    if intv_mode == "reason":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].reason_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypodesc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_imp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                    elif intv_mode == "hypofunc":
                        intv_prompts.append(batched_intv_datapoint['datapoint'].func_exp_prompt_template.format(
                            observations = intv_datapoint
                        ))
                else:
                    intv_prompts.append(batched_intv_datapoint['datapoint'].intv_prompt_template.format(
                        observations = intv_datapoint
                    ))
        intv_responses, _ = agent.run(intv_prompts)
        if isinstance(intv_responses, str):
            intv_responses = [intv_responses] 
        for idx, intv_response in enumerate(intv_responses):
            intv_lm_responses[idx].append(intv_response)
        for idx, intv_response in enumerate(intv_responses):
            if intv_mode == "intv":
                basket1, basket2 = extract_baskets(intv_response)
                intv_data = batched_intv_datapoints[idx]['blackbox'].run_ces(basket_quantity1=basket1, basket_quantity2=basket2, mode="intervention")
                intv_datapoints[idx] = intv_datapoints[idx] + "\n" + f"Basket1: {basket1}\nBasket2: {basket2}\n{intv_data}"
            elif intv_mode == "reason" or intv_mode == "hypodesc" or intv_mode == "hypofunc":
                if nintv_left[idx] % 5 == 0:
                    intv_datapoints[idx] = intv_datapoints[idx] + "\n" + intv_response
                try:
                    
                    basket1, basket2 = extract_baskets(intv_response)
                    intv_data = batched_intv_datapoints[idx]['blackbox'].run_ces(basket_quantity1=basket1, basket_quantity2=basket2, mode="intervention")
                    intv_datapoints[idx] = intv_datapoints[idx] + "\n" + f"Basket1: {basket1}\nBasket2: {basket2}\n{intv_data}"
                except Exception as e:
                    if "stop" in intv_response:
                        get_intv = False
                        print(f"Stopping intervention generation for {batched_intv_datapoints[idx]['blackbox_name']}")
                        break
                    else:
                        print(f"Error extracting baskets: {e}")
                        
                        get_intv = False
                        break
        nintv_left = [nintv_left - 1 for nintv_left in nintv_left]
        if all(x == 0 for x in nintv_left):
            get_intv = False
    return intv_lm_responses, intv_datapoints


INT_TO_BLACKBOX_NAME = {
    get_languages: ["languages"],
    get_programs: ["programs"],
    get_ces: ["ces"],
}


class InterventionMapping:
    def __init__(self):
        self.blackbox_name_to_int = {
            blackbox_name: intervention
            for intervention, blackbox_names in INT_TO_BLACKBOX_NAME.items()
            for blackbox_name in blackbox_names
        }
        
    def __call__(self, blackbox_name, intv_datapoints, agent, **kwargs):
        return self.blackbox_name_to_int[blackbox_name](intv_datapoints, agent, **kwargs)
