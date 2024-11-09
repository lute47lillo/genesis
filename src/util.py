import argparse
import networkx as nx
import numpy as np
import torch
import os
import pandas as pd
import benchmark_factory as bf
from sympy import symbols, sympify, simplify, expand
from sympy.core import Function
import sympy

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
    argparser.add_argument('--pop_size', type=int, help='Population Size (could be used as fixed parameter in many settings)', default=100)
    argparser.add_argument('--mutation_rate', type=float, help='Mutation Rate (could be used as fixed parameter in many settings)', default=0.01)
    argparser.add_argument('--inbred_threshold', type=int, help='Inbreeding Threshold. Below threshold is considered inbreeding. \
                            Minimum genetic (Hamming) distance required to allow mating', default=5)
    argparser.add_argument('--tournament_size', type=int, help='Nº of individuals to take part in the Tournament selection', default=3)
    argparser.add_argument('--exp_num_runs', type=int, help='Nº of experimental runs. (Fixed hyperparameters)', default=5)
    argparser.add_argument('--dimensions', type=int, help='GA dimensions. Used in Rugged benchmarks like MPL', default=100) 
    argparser.add_argument('--collapse_threshold', type=int, help='TODO', default=0.2) 
    argparser.add_argument('--collapse_fraction', type=int, help='TODO', default=0.1) 
    argparser.add_argument('--mpl_shift_interval', type=int, help='nº of generations for shifting global maximum in the MovingPeaksLandscape', default=30) 
    
    # Novelty Archive hyperparameters
    argparser.add_argument('--archive_nn', type=int, help='Number of nearest neighbors to consider for novelty calculation', default=20)
    argparser.add_argument('--archive_threshold', type=int, help='Minimum distance threshold for considering behaviors as novel', default=0.1) 
    argparser.add_argument('--fit_weight', type=float, help='Weight given to pure-fitness calculating individual total fitness', default=1.0)
    argparser.add_argument('--novelty_weight', type=float, help='Weight given to novelty calculating individual total fitness', default=1.0)
    
    # Genetic Programming variables
    argparser.add_argument('--max_depth', type=int, help='GP Tree maximum depth', default=15)
    argparser.add_argument('--initial_depth', type=int, help='GP Tree maximum depth', default=6)
    
    # Rugged arguments
    argparser.add_argument('--generations', type=int, help='Nº of generations to run the GA. (Used interchangeably with args.dimensions in this case)', default=100)
    argparser.add_argument('--N_NKlandscape', type=int, help='Genome Length (N) value', default=100)
    argparser.add_argument('--K_NKlandscape', type=int, help='Nº of interactions per loci (K) value', default=14)
    argparser.add_argument('--max_kinship', type=float, help='Maximal Inbreeding prevention ratio', default=0.5)
    
    # Parse all arguments
    args = argparser.parse_args()
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    return args

# ---------------- NK Landscape visualization helper functions --------------------- #

def build_lineage_graph(lineage_data):
    G = nx.DiGraph()
    for data in lineage_data:
        ind_id = data['id']
        generation = data['generation']
        ancestors = data['ancestors']
        # Add node for the individual
        G.add_node(ind_id, generation=generation)
        # Add edges from ancestors to the individual
        for ancestor_id in ancestors:
            G.add_edge(ancestor_id, ind_id)
    return G

def get_lineage_frequency(lineage_data):
    df = pd.DataFrame(lineage_data)
    # Explode the ancestors set into individual ancestor IDs
    df = df.explode('ancestors')
    # Group by generation and ancestor ID
    frequency = df.groupby(['generation', 'ancestors']).size().reset_index(name='count')
    return frequency

def save_accuracy(array, file_path_temp):
    file_path = f"{os.getcwd()}/saved_data/" + file_path_temp
    mode = 'wb'  # Write mode (overwrite or create if it doesn't exist)
    with open(file_path, mode) as f:
        np.save(f, array)
    print(f"\nAccuracy data saved to {file_path}")


# ---------------- Genetic Programming helper functions --------------------- #

def get_function_bounds(benchmark):
    
    if benchmark == 'ackley':
        bounds = (-32.768, 32.768)
    elif benchmark == 'rastrigin' or benchmark == 'sphere':
        bounds = (-5.12, 5.12)
    elif benchmark == 'rosenbrock':
        bounds = (-2.048, 2.048)
    elif benchmark == 'schwefel':
        bounds = (-500, 500)
    elif benchmark == 'griewank':
        bounds = (-600, 600)   
    elif benchmark == 'nguyen1' or benchmark == 'nguyen2' or benchmark == 'nguyen3' or benchmark == 'nguyen4' or benchmark == 'nguyen5' or benchmark == 'nguyen6':
        bounds = (-4.0, 4.0)
    elif benchmark == 'nguyen7' or benchmark == 'nguyen8':
        bounds = (0.0, 8.0)
    
    return bounds

def select_benchmark(args):
    """
        Definition
        -----------
            Select the benchmark function of Optimization class.
    """
    benchmarks = {"ackley": bf.ackley_function, "rosenbrock":bf.rosenbrock_function,
                  "rastrigin": bf.rastrigin_function, "schwefel": bf.schwefel_function,
                  "griewank" :bf.griewank_function, "sphere": bf.sphere_function}
    
    # Get function
    landscape_fn = benchmarks.get(args.benchmark)
    
    # Set the specific bounds
    args.bounds = get_function_bounds(args.benchmark)
    
    return landscape_fn

def get_function_arity(function):
    arity_dict = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        'sin': 1,
        'cos': 1,
        'log': 1
    }
    return arity_dict.get(function, 0)

def set_config_parameters(benchmark):
    
    # Experiment Parameters
    pop_sizes = [25, 50, 100, 200]
    dimensions = 10
    bounds = get_function_bounds(benchmark)
    generations = 200
    mutation_rate = 0.2
    allowed_distance = 1.0
    
    return pop_sizes, dimensions, bounds, generations, mutation_rate, allowed_distance

# --------------------------- Genetic Programming visualization ------------------------------- #

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
    
# ---------------------- Rugged Landscape Functions ------------------------- #

def extract_behavior(genome, landscape):
    
    # Get all distances
    distances = [np.sum(genome != peak.position) for peak in landscape.peaks]
    
    # Get the minimum distance to any peak for getting closest peak
    min_distance = min(distances)
    closest_peak_index = distances.index(min_distance)
    
    # Normalize over the genome length
    normalized_distance = min_distance / landscape.n  
    behavior = (closest_peak_index, normalized_distance)

    return behavior

def behavior_distance(b1, b2):
    """
        Definition
        -----------
            Get the behavioral distance between 2 different behaviors of individuals. Maximum is 2.
            
        Parameters
        -----------
            - b1 and b2 (tuples): (closest_peak_index, normalized_distance)
    """
    
    peak_index_distance = 0 if b1[0] == b2[0] else 1
    distance_difference = abs(b1[1] - b2[1])
    
    # You can weight the components differently if needed
    return peak_index_distance + distance_difference