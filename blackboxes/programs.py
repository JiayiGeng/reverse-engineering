"""
python -m blackboxes.programs
"""
import math
import json
import yaml
import random
from rich import print
from pathlib import Path
import numpy as np
import random
from typing import Any, Dict, List
from tqdm import tqdm
import copy
from utils import load_data_from_yaml


class Programs:
    def __init__(self, config_path = None, program_name="c001", seed=0):
        program_config = load_data_from_yaml(config_path) if config_path else load_data_from_yaml("configs/programs.yaml")
        program_config = program_config[program_name]
        self.dsl_expression = program_config['program']
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def parse_dsl_to_ast(self, dsl_expression = None):
        if not dsl_expression:
            dsl_expression = self.dsl_expression
        tokens = dsl_expression.replace('(', ' ( ').replace(')', ' ) ').split()
        def parse(tokens):
            if len(tokens) == 0:
                raise SyntaxError("Unexpected EOF")
            token = tokens.pop(0)
            if token == '(':
                L = []
                while tokens[0] != ')':
                    L.append(parse(tokens))
                    if len(tokens) == 0:
                        raise SyntaxError("Unexpected EOF")
                tokens.pop(0)  # pop off ')'
                return L
            elif token == ')':
                raise SyntaxError("Unexpected )")
            else:
                return token
        return parse(tokens)
    
    def ast_to_python(self, ast):
        if isinstance(ast, str):
            if ast == 'drop':
                # return "lambda x: x[1:]"
                return "(lambda n, x: x[n:])"
            
            if ast == 'droplast':
                # return "lambda x: x[:-1]"
                return "(lambda n, x: x[:-n])"
            
            if ast == '$0':
                return "x"
            
            if ast == '$1':
                return "y"
            
            if ast.isdigit():
                return ast
            
            if ast == 'empty':
                return "[]"
            return ast
        if not ast:
            return "[]"
        
        head = ast[0]
        if isinstance(head, list):
            func_code = self.ast_to_python(head)
            args_code = ', '.join(self.ast_to_python(arg) for arg in ast[1:])
            return f"{func_code}({args_code})"
        
        if head == 'lambda':  # (lambda (expr))
            body_ast = ast[1]
            if isinstance(body_ast, list) and len(body_ast) > 0:
                all_exprs_are_lists = all(isinstance(item, list) for item in body_ast)
                if all_exprs_are_lists:
                    expr_count = len(body_ast)
                    if expr_count == 1:
                        single_code = self.ast_to_python(body_ast[0])
                        return f"(lambda x: {single_code})"
                    else:
                        codes = []
                        for expr_ast in body_ast[:-1]:
                            c = self.ast_to_python(expr_ast)
                            codes.append(f"_={c}")
                        last_c = self.ast_to_python(body_ast[-1])
                        codes.append(last_c)
                        joined = ", ".join(codes)
                        return f"(lambda x: (lambda: ({joined}))())"
                else:
                    return f"(lambda x: {self.ast_to_python(body_ast)})"
            else:
                return f"(lambda x: {self.ast_to_python(body_ast)})"

        elif head == 'filter': # (filter (lambda (expr)) (list))
            _, lambda_ast, list_ast = ast
            lambda_code = self.ast_to_python(lambda_ast)
            list_code = self.ast_to_python(list_ast)
            return f"list(filter({lambda_code}, {list_code}))"
        
        elif head == 'reverse':
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)    
            return f"{list_code}[::-1]"
        
        elif head == 'append':
            _, list_ast, expr_ast = ast
            list_code = self.ast_to_python(list_ast)
            expr_code = self.ast_to_python(expr_ast)
            return f"{list_code} + [{expr_code}]"
        
        # elif head == 'filteri':
        #     _, func_ast, list_ast = ast
        #     func_code = self.ast_to_python(func_ast)
        #     list_code = self.ast_to_python(list_ast)
        # #     return f"[x for i, x in enumerate({list_code}) if {func_code}(x, i)]"
        # elif head == 'filteri':
        #     _, func_ast, list_ast = ast
        #     func_code = self.ast_to_python(func_ast)
        #     list_code = self.ast_to_python(list_ast)
        #     print(f"func_code: {func_code}")
        #     print(f"list_code: {list_code}")
        #     input()
        #     # 1-based index for DSL's second argument:
        #     return f"[x for i, x in enumerate({list_code}) if {func_code}(x, i+1)]"
        
        elif head == 'filteri':
            _, func_ast, list_ast = ast
            func_code = self.ast_to_python(func_ast)
            list_code = self.ast_to_python(list_ast)
            # Rename the loop variable to "elem" to avoid the name conflict
            return f"[elem for i, elem in enumerate({list_code}) if {func_code}(elem, i+1)]"
                
        elif head == 'map':
            _, func_ast, list_ast = ast
            func_code = self.ast_to_python(func_ast)
            list_code = self.ast_to_python(list_ast)
            return f"list(map({func_code}, {list_code}))"
        
        elif head == 'mapi':
            _, op_ast, list_ast = ast
            op_code = self.ast_to_python(op_ast)
            list_code = self.ast_to_python(list_ast)
            
            if op_code == '+':
                return f"[({list_code}[i] + (i+1)) for i in range(len({list_code}))]"
            else:
                raise NotImplementedError(f"Unknown operation: {op_code}")
        
        elif head == 'sum':
            # AST: ["sum", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"sum({list_code})"
        
        elif head == '+':
            # AST: ["+", lhs_ast, rhs_ast]
            _, lhs_ast, rhs_ast = ast
            lhs_code = self.ast_to_python(lhs_ast)
            rhs_code = self.ast_to_python(rhs_ast)
            return f"({lhs_code}) + ({rhs_code})"
        
        elif head == 'max':
            # AST: ["max", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"max({list_code})"
        
        elif head == 'min':
            # AST: ["min", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"min({list_code})"
        
        elif head == 'last':
            # AST: ["last", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[-1]"
        
        elif head == 'droplast':
            _, n_ast, list_ast = ast
            n_code = self.ast_to_python(n_ast)      
            list_code = self.ast_to_python(list_ast) 
            return f"{list_code}[:-{n_code}]"
        
        elif head == 'repeat':
            _, item_ast, count_ast = ast
            item_code = self.ast_to_python(item_ast)
            count_code = self.ast_to_python(count_ast)
            return f"[{item_code}] * {count_code}"
        
        # elif head == 'replace':
        #     # AST: ["replace", idx_ast, expr_ast, list_ast]
        #     _, idx_ast, expr_ast, list_ast = ast
        #     idx_code = self.ast_to_python(idx_ast)
        #     expr_code = self.ast_to_python(expr_ast)
        #     list_code = self.ast_to_python(list_ast)
        #     return f"{list_code}[:({idx_code})] + [{expr_code}] + {list_code}[({idx_code})+1:]"
        elif head == 'replace':
            # AST: ["replace", idx_ast, expr_ast, list_ast]
            _, idx_ast, expr_ast, list_ast = ast
            idx_code = self.ast_to_python(idx_ast)
            expr_code = self.ast_to_python(expr_ast)
            list_code = self.ast_to_python(list_ast)

            # 1-based -> 0-based with out-of-range check
            return (
                f"(lambda __lst: "
                f"__lst[:(({idx_code})-1)] + [{expr_code}] + __lst[(({idx_code})-1)+1:] "
                f"if 0 <= (({idx_code})-1) < len(__lst) "
                f"else __lst"
                f")({list_code})"
            )
                
        elif head == 'cons':
            # AST: ["cons", expr_ast, list_ast]
            _, expr_ast, list_ast = ast
            expr_code = self.ast_to_python(expr_ast)
            list_code = self.ast_to_python(list_ast)
            return f"[{expr_code}] + {list_code}"
        
        elif head == 'if':
            # AST: ["if", cond_ast, then_ast, else_ast]
            _, cond_ast, then_ast, else_ast = ast
            cond_code = self.ast_to_python(cond_ast) 
            then_code = self.ast_to_python(then_ast) 
            else_code = self.ast_to_python(else_ast) 
            return f"( ({then_code}) if ({cond_code}) else ({else_code}) )"
        
        elif head == '>':
            # AST: [">", lhs_ast, rhs_ast]
            _, lhs_ast, rhs_ast = ast
            lhs_code = self.ast_to_python(lhs_ast)
            rhs_code = self.ast_to_python(rhs_ast)
            return f"({lhs_code}) > ({rhs_code})"
        
        elif head == '<':
            # AST: ["<", lhs_ast, rhs_ast]
            _, lhs_ast, rhs_ast = ast
            lhs_code = self.ast_to_python(lhs_ast)
            rhs_code = self.ast_to_python(rhs_ast)
            return f"({lhs_code}) < ({rhs_code})"
        
        elif head == '==':
            # AST: ["==", lhs_ast, rhs_ast]
            _, lhs_ast, rhs_ast = ast
            lhs_code = self.ast_to_python(lhs_ast)
            rhs_code = self.ast_to_python(rhs_ast)
            return f"({lhs_code}) == ({rhs_code})"
        
        elif head == 'length':
            # AST: ["length", <some_list>]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"len({list_code})"
        
        elif head == 'singleton':
            # AST: ["singleton", expr_ast]
            expr_ast = ast[1]
            expr_code = self.ast_to_python(expr_ast)
            return f"[{expr_code}]"
        
        elif head == 'first':
            # AST: ["first", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[0]"
        
        elif head == 'second':
            # AST: ["second", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[1]"
        
        elif head == 'third':
            # AST: ["third", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[2]"
        
        elif head == 'swap':
            _, i_ast, j_ast, list_ast = ast
            i_code = self.ast_to_python(i_ast)
            j_code = self.ast_to_python(j_ast)
            list_code = self.ast_to_python(list_ast)
            return (
                    f"(lambda __l: ["
                    f"__l[k] if k not in (({i_code})-1, ({j_code})-1) "
                    f"else (__l[({j_code})-1] if k == ({i_code})-1 else __l[({i_code})-1]) "
                    f"for k in range(len(__l))]"
                    f")({list_code})"
                )
        
        elif head == 'nth':
            # AST: ["nth", i_ast, list_ast]
            _, i_ast, list_ast = ast
            i_code = self.ast_to_python(i_ast)
            list_code = self.ast_to_python(list_ast)
            # return f"{list_code}[{i_code}]"
            return f"{list_code}[({i_code}) - 1]"
        
        elif head == 'take':
            # AST: ["take", n_ast, list_ast]
            _, n_ast, list_ast = ast
            n_code = self.ast_to_python(n_ast)
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[:{n_code}]"
        
        elif head == 'drop':
            # AST: ["drop", n_ast, list_ast]
            _, n_ast, list_ast = ast
            n_code = self.ast_to_python(n_ast)
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[{n_code}:]"
        
        elif head == 'flatten':
            # AST: ["flatten", list_ast]
            list_ast = ast[1]
            list_code = self.ast_to_python(list_ast)
            return f"[item for sublist in {list_code} for item in sublist]"
        
        elif head == 'slice':
            # AST: ["slice", start_ast, end_ast, list_ast]
            _, start_ast, end_ast, list_ast = ast
            start_code = self.ast_to_python(start_ast)
            end_code = self.ast_to_python(end_ast)
            list_code = self.ast_to_python(list_ast)
            return f"{list_code}[{start_code} - 1 : {end_code}]"
        
        elif head == 'concat':
            # AST: ["concat", list1_ast, list2_ast]
            _, list1_ast, list2_ast = ast
            list1_code = self.ast_to_python(list1_ast)
            list2_code = self.ast_to_python(list2_ast)
            return f"{list1_code} + {list2_code}"  
        
        elif head == 'is_in':
            # AST: ["is_in", expr_ast, list_ast]
            _, expr_ast, list_ast = ast
            expr_code = self.ast_to_python(expr_ast)
            list_code = self.ast_to_python(list_ast)
            return f"({expr_code}) in ({list_code})"
        
        elif head == 'is_odd':
            # AST: ["is_odd", expr_ast]
            _, expr_ast = ast
            expr_code = self.ast_to_python(expr_ast)
            return f"({expr_code}) % 2 == 1"
 
        elif head == 'cut_idx':
            _, idx_ast, list_ast = ast
            idx_code = self.ast_to_python(idx_ast)
            list_code = self.ast_to_python(list_ast)
            # 1-based → 0-based
            return (
                f"(lambda __lst: "
                f"__lst[:(({idx_code}) - 1)] + __lst[(({idx_code}) - 1) + 1:] "
                f")({list_code})"
            )

        elif head == 'insert':
            # DSL: (insert <element> <1-based index> <list>)
            _, elem_ast, idx_ast, list_ast = ast

            elem_code = self.ast_to_python(elem_ast)
            idx_code = self.ast_to_python(idx_ast)
            list_code = self.ast_to_python(list_ast)
            
            # Insert `elem_code` at position (idx_code - 1) in Python
            return (
                f"(lambda __lst: "
                f"__lst[:(({idx_code}) - 1)] + "
                f"[{elem_code}] + "
                f"__lst[(({idx_code}) - 1):]"
                f")({list_code})"
            )
        else:
            raise NotImplementedError(f"Unknown operation head: {head}")
    
    def compile_dsl(self, dsl_expression = None):
        if not dsl_expression:
            dsl_expression = self.dsl_expression
        # Parse the DSL expression to AST and convert it to Python code
        ast = self.parse_dsl_to_ast(dsl_expression)
        py_code = self.ast_to_python(ast)
        compiled_obj = eval(py_code)
        return compiled_obj
        
    def run_program(self, input_list, output_list = None, dsl_expression = None, mode = "observation"):
        if not dsl_expression:
            dsl_expression = self.dsl_expression
        
        if mode == "observation":
            try:
                output = self.compile_dsl(dsl_expression)(input_list)
                # check if output is function
                if callable(output): # Dealing with the drop and droplast functions
                    output = output(input_list)
                return output
            except:
                return f"invalid input"
        
        elif mode == "intervention":        
            try:
                output = self.compile_dsl(dsl_expression)(input_list)
                if output_list == "none" or not output_list:
                    return output
                else:
                    if output == output_list:
                        output_string = " ".join(map(str, output_list))
                        return f"{output_string} => Correct"
                    else:
                        output_string = " ".join(map(str, output_list))
                        return f"{output_string} => Incorrect"
            except:
                return f"invalid input"
            

def flatten(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def find_max_dollar_index(ast) -> int:
    if isinstance(ast, str):
        if ast.startswith('$') and ast[1:].isdigit():
            return int(ast[1:])
        else:
            return -1
    elif isinstance(ast, list):
        mx = -1
        for elem in ast:
            sub = find_max_dollar_index(elem)
            if sub > mx:
                mx = sub
        return mx
    else:
        return -1
    

def swap(lst, i, j):
    i -= 1
    j -= 1
    
    new_lst = list(lst)
    new_lst[i], new_lst[j] = new_lst[j], new_lst[i]
    return new_lst

if __name__ == "__main__":
    print(f"tests")
  
    program = Programs(program_name="c095")
    input_list = [1, 2, 3, 4, 5]
    # Test intervention mode with output_list
    output_list = [2, 4, 3, 2]
    output = program.run_program(input_list = input_list, output_list = output_list, mode = "intervention")
    print(f"output: {output}")
    
   