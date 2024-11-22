import argparse
import numpy as np
import torch
import os
import pandas as pd
import benchmark_factory as bf
from sympy import symbols, sympify, simplify, expand
import sympy
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def set_args():
    
    # Define arguments
    argparser = argparse.ArgumentParser()
    
    # Global Experimental variables
    argparser.add_argument('--seed', type=int, help='Seed for random', default=99)
    argparser.add_argument('--device', type=str, help='CPU/GPU usage', default="cpu")
    argparser.add_argument('--benchmark', type=str, help='Optimization function to run', default="rastrigin")
    argparser.add_argument('--bench_name', type=str, help='Problem landscape name (ie: MovingPeaksLandscape)', default="none")
    argparser.add_argument('--config_plot', type=str, help='plot info details', default="none")
    argparser.add_argument('--current_run', type=int, help='Current experimental run if multiple runs', default=0)
    
    # Experimental Hyperparameters
    argparser.add_argument('--generations', type=int, help='Nº of generations to run the GA.', default=100)
    argparser.add_argument('--pop_size', type=int, help='Population Size (could be used as fixed parameter in many settings)', default=100)
    argparser.add_argument('--mutation_rate', type=float, help='Mutation Rate (could be used as fixed parameter in many settings)', default=0.01)
    argparser.add_argument('--inbred_threshold', type=int, help='Inbreeding Threshold. Below threshold is considered inbreeding. \
                            Minimum genetic (tree-edit) distance required to allow mating', default=5)
    argparser.add_argument('--tournament_size', type=int, help='Nº of individuals to take part in the Tournament selection', default=3)
    argparser.add_argument('--exp_num_runs', type=int, help='Nº of experimental runs. (Fixed hyperparameters)', default=5)
    
    # Genetic Programming variables
    argparser.add_argument('--max_depth', type=int, help='GP Tree maximum depth', default=15)
    argparser.add_argument('--initial_depth', type=int, help='GP Tree maximum depth', default=6)
    
    # Parse all arguments
    args = argparser.parse_args()
    
    # Set the seed for reproducibility
    args.seed = random.randint(0, 999999)
    set_seed(args.seed)
    
    return args

# ---------------- Write/Read helper functions --------------------- #

def save_accuracy(array, file_path_temp):
    file_path = f"{os.getcwd()}/saved_data/" + file_path_temp
    mode = 'wb'  # Write mode (overwrite or create if it doesn't exist)
    with open(file_path, mode) as f:
        np.save(f, array)
    print(f"\nAccuracy data saved to {file_path}")

# ---------------- Genetic Programming helper functions --------------------- #

def get_function_bounds(benchmark):
    
    if benchmark == 'nguyen1' or benchmark == 'nguyen2' or benchmark == 'nguyen3' or benchmark == 'nguyen4' or benchmark == 'nguyen5' or benchmark == 'nguyen6':
        bounds = (-4.0, 4.0)
    elif benchmark == 'nguyen7' or benchmark == 'nguyen8':
        bounds = (0.0, 8.0)
    
    return bounds

def select_gp_benchmark(args):
    """
        Definition
        -----------
            Select the benchmark for Genetic Programming experiments.
    """
    benchmarks = {"nguyen1": bf.nguyen1, "nguyen2": bf.nguyen2,
                  "nguyen3": bf.nguyen3, "nguyen4": bf.nguyen4,
                  "nguyen5": bf.nguyen5, "nguyen6": bf.nguyen6,
                  "nguyen7": bf.nguyen7, "nguyen8": bf.nguyen8}
    
    # Get function
    gp_bench_fn = benchmarks.get(args.benchmark)
    
    # Set the specific bounds
    args.bounds = get_function_bounds(args.benchmark)
    
    return gp_bench_fn

# --------------------------- Genetic Programming visualization ------------------------------- #

class Node:
    def __init__(self, value, children=None):
        self.value = value  # Operator (e.g., '+', '-', '*', '/') or Operand (e.g., 'x', '1.0')
        self.children = children if children is not None else []

    def is_terminal(self):
        """
        Determines if the node is a terminal node (no children).
        """
        return len(self.children) == 0

    def __repr__(self):
        return f"Node({self.value})"

def tokenize(expression):
    """
    Converts the input string into a list of tokens.
    """
    tokens = []
    current_token = ""
    for char in expression:
        if char in ('(', ')'):
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        elif char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def parse(tokens):
    """
    Parses the list of tokens into a tree of Nodes.
    """
    if len(tokens) == 0:
        return None, tokens

    token = tokens.pop(0)

    if token == '(':
        # Next token should be the operator
        if len(tokens) == 0:
            raise SyntaxError("Unexpected end of tokens after '('")
        operator = tokens.pop(0)
        node = Node(operator)
        while tokens and tokens[0] != ')':
            child, tokens = parse(tokens)
            if child is not None:
                node.children.append(child)
        if not tokens:
            raise SyntaxError("Missing ')' in expression")
        tokens.pop(0)  # Remove ')'
        return node, tokens
    elif token == ')':
        # Should not reach here
        raise SyntaxError("Unexpected ')' in expression")
    else:
        # Operand or operator without children (terminal node)
        return Node(token), tokens


def print_tree(node, indent=""):
    """
    Recursively prints the tree in an indented format.
    """
    if node is None:
        print(indent + "None")
        return
    print(indent + str(node.value))
    for child in node.children:
        print_tree(child, indent + "  ")
        
def tree_to_expression(node):
    """
    Converts the Node tree into a simplified SymPy expression.
    """
    sympy_expr = tree_to_sympy(node)
    expanded_expr = expand(sympy_expr)
    simplified_expr = simplify(expanded_expr)
    return simplified_expr
        
def convert_tree_to_expression(expression_str):
    """
    Converts a GP tree string in prefix notation into a simplified SymPy expression.
    
    Parameters:
        expression_str (str): The GP tree in prefix notation.
    
    Returns:
        sympy.Expr: The simplified SymPy expression.
    """
    tokens = tokenize(expression_str)
    try:
        tree, remaining = parse(tokens)
        if remaining:
            print("Warning: Remaining tokens after parsing:", remaining)
    except SyntaxError as e:
        print("Syntax Error during parsing:", e)
        tree = None
    
    if tree is None:
        raise ValueError("Invalid expression string. Parsing failed.")
    
    expr = tree_to_expression(tree)
    return expr

def tree_to_sympy(node):
    """
    Recursively converts a Node tree into a SymPy expression.
    """
    # Define the symbol 'x'
    x = symbols('x')    
    
    if node.is_terminal():
        if node.value == 'x':
            return x
        else:
            try:
                return sympify(node.value)
            except:
                raise ValueError(f"Invalid terminal node value: {node.value}")
    else:
        func = node.value
        args = [tree_to_sympy(child) for child in node.children]
        if func == '+':
            return args[0] + args[1]
        elif func == '-':
            if len(args) == 1:
                return -args[0]
            elif len(args) == 2:
                return args[0] - args[1]
            else:
                raise ValueError(f"Unsupported number of arguments for '-': {len(args)}")
        elif func == '*':
            return args[0] * args[1]
        elif func == '/':
            return args[0] / args[1]
        elif func == 'sin':
            return sympy.sin(args[0])
        elif func == 'cos':
            return sympy.cos(args[0])
        elif func == 'log':
            return sympy.log(args[0])
        else:
            raise ValueError(f"Unsupported function: {func}")
        
# -------------- Plotting helper functions --------------- #
        
def create_padded_df(data, metric, run_ids):
    
    # Determine the maximum length for the metric across all runs
    max_length = max(len(run_data[metric]) for run_data in data.values())
    
    # Initialize a DataFrame with NaNs
    df = pd.DataFrame(index=range(max_length), columns=run_ids, dtype=float)
    
    # Populate the DataFrame
    for run_id, run_data in data.items():
        values = run_data[metric]
        df[run_id].iloc[:len(values)] = values
    
    return df

def pad_sublist(sublist, target_length):
    """
    Pads the sublist to the target_length by repeating the last element.
    
    Parameters:
    - sublist (list): The original sublist.
    - target_length (int): The desired length after padding.
    
    Returns:
    - list: The padded sublist.
    """
    current_length = len(sublist)
    if current_length < target_length:
        padding = [sublist[-1]] * (target_length - current_length)
        return sublist + padding
    else:
        return sublist
    
# Flatten the Data
def flatten_results_depths(treatment_name, depths):
    data_df = []
    for depth in depths:
        # Load dict data
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/gp_lambda/PopSize:300_InThres:4_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:3_{treatment_name}.npy"
        data = np.load(file_path_name, allow_pickle=True)
        data_dict = data.item()
        
        # Iterate over and create DataFram
        for run, metrics in data_dict.items():
            # diversity = metrics['diversity']# TODO: For another plto
            generation_success = metrics['generation_success']
            data_df.append({
                'Treatment': treatment_name,
                'Depth': depth,
                'Run': run,
                'Generation_Success': generation_success
            })
    return pd.DataFrame(data_df)

# Flatten the Data
def flatten_results_thresholds(treatment_name, thresholds):
    data_df = []
    for thres in thresholds:
        # Load dict data
        file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/nguyen2/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:9_InitD:3_{treatment_name}.npy"
        data = np.load(file_path_name, allow_pickle=True)
        data_dict = data.item()
        
        # Iterate over and create DataFram
        for run, metrics in data_dict.items():
            # diversity = metrics['diversity']# TODO: For another plto
            generation_success = metrics['generation_success']
            data_df.append({
                'Treatment': treatment_name,
                'Thresholds': thres,
                'Run': run,
                'Generation_Success': generation_success
            })
    return pd.DataFrame(data_df)

def flatten_results_in_max_depth_diversity(bench_name, treatment_name, thresholds, depths, init_depth):
    data_df = []
    for thres in thresholds:
        for depth in depths:
            # Load dict data
            file_path_name = f"{os.getcwd()}/saved_data/genetic_programming/{bench_name}/gp_lambda/PopSize:300_InThres:{thres}_Mrates:0.0005_Gens:150_TourSize:15_MaxD:{depth}_InitD:{init_depth}_{treatment_name}.npy"
            data = np.load(file_path_name, allow_pickle=True)
            data_dict = data.item()
            for run, metrics in data_dict.items():
                # TODO: If Wanted al values needs to padd:
                # padded_diversity = [pad_sublist(sublist, target_length) for sublist in metrics['diversity']]
                # metrics['diversity'] = padded_diversity
                diversity = metrics['diversity'][-1]
                generation_success = metrics['generation_success']
        
                # Update
                data_df.append({
                    'Treatment': treatment_name,
                    'Max_Depth': depth,
                    'Inbred_Threshold': thres,
                    'Run': run,
                    'Generation_Success': generation_success,
                    'Diversity': diversity
                })
    return pd.DataFrame(data_df)

# Padding Function
def pad_sublist(sublist, target_length):
    current_length = len(sublist)
    if current_length < target_length:
        padding = [sublist[-1]] * (target_length - current_length)
        return sublist + padding
    else:
        return sublist

# Determine Global Maximum Depth
def get_global_max_depth(*results_dicts):
    max_depth = 0
    for results in results_dicts:
        for run in results:
            current_depth = len(results[run]['diversity'])
            if current_depth > max_depth:
                max_depth = current_depth
    return max_depth

# Pad 'diversity' Lists
def pad_diversity_lists(results_dict, target_length):
    for run in results_dict:
        original_length = len(results_dict[run]['diversity'])
        padded_diversity = [pad_sublist(sublist, target_length) for sublist in results_dict[run]['diversity']]
        results_dict[run]['diversity'] = padded_diversity
        print(f"Run {run}: Padded Diversity Lengths = {[len(s) for s in results_dict[run]['diversity']]}")
    return results_dict