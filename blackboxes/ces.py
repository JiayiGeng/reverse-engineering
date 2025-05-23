"""
python -m blackboxes.ces
"""
import math
import json
import yaml
import random
import copy
from rich import print
from pathlib import Path
import numpy as np
import random
from typing import Any, Dict, List
from tqdm import tqdm
import copy
from utils import load_data_from_yaml


class CES:
    def __init__(self, config_path = None, ces_name="ces_1"):
        ces_config = load_data_from_yaml(config_path) if config_path else load_data_from_yaml("configs/ces.yaml")
        ces_config = ces_config[ces_name]
        self.baskets = []
        for idx, name in enumerate(ces_config['names']):
            self.baskets.append({"obj": name, "alpha": ces_config['alphas'][idx]})
        self.rho = ces_config['rho']
        self.quantity_of_goods = ces_config['quantity_of_goods']

    def compute_utility(self,rho, quantity):
        total_sum = 0.0
        basket = copy.deepcopy(self.baskets)
        for idx, item in enumerate(basket):
            # Update the basket with the actual quantity
            item["quantity"] = quantity[idx]
            # Accumulate alpha_i * (x_i ^ rho)
            total_sum += item["alpha"] * (quantity[idx] ** self.rho)

        # Compute the final CES utility
        utility = total_sum ** (1.0 / self.rho)
        return utility
    
    def run_ces(self, basket_quantity1 = "none", basket_quantity2 = "none", estimate_preference = "none", mode = "observation"):
        if mode == "observation":
            # randomly sample two lists of quantity
            # basket_quantity1 = random.sample(range(1, 100), self.quantity_of_goods)
            # basket_quantity2 = random.sample(range(1, 100), self.quantity_of_goods)
            basket_quantity1 = [random.uniform(1, 100) for _ in range(self.quantity_of_goods)]
            basket_quantity2 = [random.uniform(1, 100) for _ in range(self.quantity_of_goods)]
            basket_utility1 = self.compute_utility(self.rho, basket_quantity1)
            basket_utility2 = self.compute_utility(self.rho, basket_quantity2)
            if basket_utility1 > basket_utility2:
                preference = f"Preference: Basket1"
            elif basket_utility1 < basket_utility2:
                preference = f"Preference: Basket2"
            else:
                preference = f"Preference: equal utility"
            return preference, basket_quantity1, basket_quantity2
        
        elif mode == "intervention":
            if len(basket_quantity1) != self.quantity_of_goods or len(basket_quantity2) != self.quantity_of_goods:
                return "invalid quantity in the basket"
            if estimate_preference != "none":
                if basket_quantity1 == "none" or basket_quantity2 == "none":
                    return "invalid query"
                else:
                    basket_utility1 = self.compute_utility(self.rho, basket_quantity1)
                    basket_utility2 = self.compute_utility(self.rho, basket_quantity2)
                    if basket_utility1 > basket_utility2:
                        if estimate_preference.lower() == "basket1":
                            return "correct"
                        else:
                            return "incorrect"
                    elif basket_utility1 < basket_utility2:
                        if estimate_preference.lower() == "basket2":
                            return "correct"
                        else:
                            return "incorrect"
                    else:
                        if "equal" in estimate_preference.lower():
                            return "correct"
                        else:
                            return "incorrect"
            else:
                if basket_quantity1 == "none" or basket_quantity2 == "none":
                    return "invalid query"
                else:
                    basket_utility1 = self.compute_utility(self.rho, basket_quantity1)
                    basket_utility2 = self.compute_utility(self.rho, basket_quantity2)
                    
                    if basket_utility1 > basket_utility2:
                        return "Preference: Basket1"
                    elif basket_utility1 < basket_utility2:
                        return "Preference: Basket2"
                    else:
                        return "Preference: equal utility"
        
        

if __name__ == "__main__":
    ces = CES()
    quantity = [2, 10]
    rho = 0.5
    basket_quantity1 = [2, 10]
    basket_quantity2 = [1, 10]
    estimate_preference = "basket1"
    result = ces.run_ces(basket_quantity1, basket_quantity2, estimate_preference, mode = "intervention")
    print(result)