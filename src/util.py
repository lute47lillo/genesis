import argparse
import networkx as nx
import numpy as np
import torch
import pandas as pd

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
    argparser.add_argument('--dimensions', type=int, help='GA dimensions', default=10) 
    
    # Genetic Programming variables
    argparser.add_argument('--max_depth', type=int, help='GP Tree maximum depth', default=15)
    argparser.add_argument('--initial_depth', type=int, help='GP Tree maximum depth', default=6)
    
    # NK-Landscape arguments
    argparser.add_argument('--generations', type=int, help='Nº of generations to run the GA. (Used interchangeably with args.dimensions in this case)', default=100)
    argparser.add_argument('--N_NKlandscape', type=int, help='Genome Length (N) value', default=100)
    argparser.add_argument('--K_NKlandscape', type=int, help='Nº of interactions per loci (K) value', default=14)
    argparser.add_argument('--pop_size', type=int, help='Population Size (could be used as fixed parameter in many settings)', default=100)
    argparser.add_argument('--mutation_rate', type=float, help='Mutation Rate (could be used as fixed parameter in many settings)', default=0.01)
    argparser.add_argument('--inbred_threshold', type=int, help='Inbreeding Threshold. Below threshold is considered inbreeding. \
                            Minimum genetic (Hamming) distance required to allow mating', default=5)
    argparser.add_argument('--tournament_size', type=int, help='Nº of individuals to take part in the Tournament selection', default=3)
    argparser.add_argument('--exp_num_runs', type=int, help='Nº of experimental runs. (Fixed hyperparameters)', default=5)

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

# ---------------- Genetic Programming helper functions --------------------- #

def get_function_bounds(benchmark):
    
    if benchmark == 'ackley':
        bounds = (-32.768, 32.768)
    elif benchmark == 'rastrigin':
        bounds = (-5.12, 5.12)
    elif benchmark == 'rosenbrock':
        bounds = (-2.048, 2.048)
    elif benchmark == 'schwefel':
        bounds = (-500, 500)
    elif benchmark == 'griewank':
        bounds = (-600, 600)   
    elif benchmark == 'sphere':
        bounds = (-5.12, 5.12)
    elif benchmark == 'nguyen1':
        bounds = (-4.0, 4.0)
    
    return bounds

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