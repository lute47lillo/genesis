import numpy as np
import copy
import math
import numpy as np
import util
import random

class NKLandscape:
    """ N-K Fitness Landscape """
    def __init__(self, args, n=10, k=2):
        self.n = n  # The number of genes (loci) in the genome. Genome length
        self.k = k  # Number of other loci interacting with each gene. For each gene 2^(k+1) possible combinations of gene states that affect its fitness contribution
        self.gene_contribution_weight_matrix = np.random.rand(n, 2**(k+1)) 
        args.bench_name = 'nk_landscape'
        
    def get_contributing_gene_values(self, genome, gene_num):
        contributing_gene_values = ""
        for i in range(self.k+1):
            contributing_gene_values += str(genome[(gene_num + i) % self.n])
        return contributing_gene_values
    
    def get_fitness(self, genome):
        """
            Higher values of K increases the epistatic interactions (dependencies among genes). 
            Therefore, making the landscape more rugged.
            
            Maximum that one gene can contribute is 1. 
            Fitness of a genome is the average over the sum of the fitness contirbution of each gene.
        """
        gene_values = np.zeros(self.n)
        for gene_num in range(len(genome)):
            contributing_gene_values = self.get_contributing_gene_values(genome, gene_num)
            index = int(contributing_gene_values, 2)
            gene_values[gene_num] = self.gene_contribution_weight_matrix[gene_num, index]
        return np.mean(gene_values)
    
# -------------- Symbolic Regression Functions --------------- #

def nguyen1(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 3 + x ** 2 + x
    return y

def nguyen2(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen3(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 5 + x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen4(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x
    return y

def nguyen5(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = np.sin(x**2) * np.cos(x) - 1
    return y

def nguyen6(x):
    """
        Input E[-4.0, 4.0] with step size of 0.1
    """
    y = np.sin(x) + np.sin(x + x ** 2)
    return y

def nguyen7(x):
    """
        Input E[0.0, 8.0] with step size of 0.1. Undefined for inputs smaller than -1
    """
    y = np.log(x+1) + np.log(x ** 2 + 1)
    return y

def nguyen8(x):
    """
        Input E[0.0, 8.0] with step size of 0.1. Undefined for negative inputs
    """
    y = np.sqrt(x)
    return y

# -------------- Optimization Functions --------------- #

# Rastrigin Function
class Rastrigin:
    def __init__(self, args,  A = 10):
        self.A = A
        args.bench_name = 'Rastrigin'
        
    def get_fitness(self, x):
        return self.A * len(x) + sum(xi**2 - self.A * np.cos(2 * np.pi * xi) for xi in x)
        
def rastrigin_function(self, x, A = 10):
    return self.A * len(x) + sum(xi**2 - self.A * np.cos(2 * np.pi * xi) for xi in x)
    
def sphere_function(x):
    """
        [-5.12, 5.12]
    """
    return sum(xi ** 2 for xi in x)

def rosenbrock_function(x):
    """
        [-2.048, 2.048]
    """
    return sum(100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    """
        # Parameters specific to Ackley Function
        bounds_ackley = (-32.768, 32.768)
    """
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.exp(1)

def rastrigin_function(x, A=10):
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def schwefel_function(x):
    """
        [-500, 500]
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

def griewank_function(x):
    """
        [-600, 600]
    """
    sum_sq = sum(xi ** 2 for xi in x) / 4000
    prod_cos = np.prod([math.cos(xi / math.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_sq - prod_cos + 1

# ------------- Classic benchmarks ----------------- #

class Jump:
    
    def __init__(self, args, k=10):
        self.k = k
        args.bench_name = 'Jump' 
        
    def get_fitness(self, genes):
        """
            Global optimum is the string with all ones. 
            Introduces a 'gap' or 'jump' needed to be crossed to get to the global minimum.
            
            k is the 'jump' size.
        """
        n = len(genes)
        u = np.sum(genes)
        if u == n:
            return n
        elif u <= n - self.k:
            return u - self.k
        else:
            return n - u
        
class JumpOffsetSpike:
    """
        Global Optimum: The string with all ones (u = n), with maximum fitness n + o.
            - o = offset 
            - n = lengh
            - k = jump size
    """
    def __init__(self, args, o=20, k=10):
        self.k = k
        self.o = o
        args.bench_name = 'JumpOffsetSpike' 
        
        
    def jump_offset_spike_fitness(self, genes):
        """
            Higher values of o and k makes the problem more difficult
        """
        n = len(genes)
        u = np.sum(genes)
        
        if u == n:
            return n + self.o
        elif u <= n - self.k:
            return u
        else:
            return (n - u) - self.o
        
class LeadingOnes:  
    def __init__(self, args):
        args.bench_name = 'LeadingOnes'  
    
    def get_fitness(self, genes):
        fitness = 0
        for gene in genes:
            if gene == 1:
                fitness += 1
            else:
                break
        return fitness

class OneMax:  
    def __init__(self, args):
        args.bench_name = 'OneMax' 
    
    def get_fitness(self, genes):
        return np.sum(genes)
    
class DeceptiveLeadingBlocks:
    """
        Global Optimum: When all blocks have 0s given we are working with binaray genome strings.
    """
    
    def __init__(self, args, block_size=5):
        self.block_size = block_size
        args.bench_name = 'DeceptiveLeadingBlocks'
    
    def deceptive_block(self, block):
        # Length of the block
        l = len(block)
        
        # Number of ones in the block
        u = np.sum(block)
        if u == l:
            return l
        else:
            return l - 1 - u
        
    def get_fitness(self, genes):
        """
            The genome is divided into blocks, and each block contributes to the overall fitness in a way that can deceive the GA.
        """
        n = len(genes)
        assert n % self.block_size == 0, "Genome length must be divisible by block size."
        fitness = 0
        num_blocks = n // self.block_size
        for i in range(num_blocks):
            block = genes[i * self.block_size : (i + 1) * self.block_size]
            fitness += self.deceptive_block(block)
        return fitness
    
# --------------- More advanced Novelty-Search / Open-end benchmarks ---------- #
class Peak:
        
    def __init__(self, position, height, width):
        self.position = np.array(position)  # Binary array
        self.height = height
        self.width = width  # Controls the steepness of the peak

class MovingPeaksLandscape:
    """
        Definition
        -----------
            Rugged Dynamic Landscape. Some key points are:
                - Individuals close to any peak have higher fitness. 
                - Fitness is based on proximity to peaks so the population needs to be diversed.
                - As a dynamic landscape, the pressure for diversity is needed to increase the likelihood 
                of having individuals close to peaks when they move.
    
    """
    def __init__(self, args, m=3, h_min=70.0, h_max=100.0, w_min=10.0, w_max=12.0):
        self.args = args
        self.n = args.dimensions  # Genome length
        self.m = m  # Number of peaks
        self.h_min = h_min
        self.h_max = h_max
        self.w_min = w_min
        self.w_max = w_max
        self.shift_interval = args.mpl_shift_interval       # Generations between shifts
        self.n_shifts = int(self.args.generations / self.args.mpl_shift_interval)
        self.peaks = []
        self.pre_compute_shifted_peaks()
        args.bench_name = 'MovingPeaksLandscape'
        
        # print(f"Initial Peaks:")
        # for idx, peak in enumerate(self.peaks):
        #     print(f"\nPeak {idx}. Position: {peak.position}.")
        #     print(f"Height: {peak.height}. Width: {peak.width}")
        
    def initialize_peaks(self):
        self.peaks = []
        for _ in range(self.m):
            position = np.random.randint(2, size=self.n)        # Random binary position
            height = np.random.uniform(self.h_min, self.h_max)
            width = np.random.uniform(self.w_min, self.w_max)
            self.peaks.append(Peak(position, height, width))
        self.update_global_optimum()
        
    def pre_compute_shifted_peaks(self):
        """
            Definition
            -----------
                Precompute the shifting effect on the Peaks in order to use the same Peaks and Landscape for both inbreeding and no inbreeding controls.
        """
        # Init peaks
        self.initialize_peaks()
        
        # Init dictionary to story the peaks at every shift with a deep copy
        temp_peaks = copy.deepcopy(self.peaks)
        
        # Initialize the pre_peaks dictionary to store data for all runs
        self.pre_peaks = {}
        
        for run in range(0, self.args.exp_num_runs):
            
            # Reinitialize the seed so every run is different
            util.set_seed(random.randint(0, 9999))
            
            # Deep copy the original peaks to avoid mutating them across runs
            temp_peaks = copy.deepcopy(self.peaks)
            self.pre_peaks[run] = {}
            
            # Base Case: Shift 0 (original peaks)
            self.pre_peaks[run][0] = {'peaks': []}
            for peak in temp_peaks:
                # It's important to deepcopy each peak if they are mutable
                self.pre_peaks[run][0]['peaks'].append(copy.deepcopy(peak))
        
            # For all shifts, pre-calculate the shifts
            for shift in range(1, self.n_shifts+1):
                # Init shift dict and create a deep copy
                self.pre_peaks[run][shift] = {'peaks': []}
                shifted_peaks = copy.deepcopy(temp_peaks)
                
                for peak in shifted_peaks:
                    
                    # Flip a random number of bits to shift the peak
                    num_bits_to_flip = np.random.randint(1, int(0.3 * self.n) + 1)  # Flip up to N% of bits. TODO: Adjust as hyperparameter for difficulty
                    flip_indices = np.random.choice(self.n, num_bits_to_flip, replace=False)
                    peak.position[flip_indices] = 1 - peak.position[flip_indices]
                    
                    # Adjust height and width
                    peak.height += np.random.normal(0, 2.5)
                    peak.height = np.clip(peak.height, self.h_min, self.h_max)
                    peak.width += np.random.normal(0, 0.5)
                    peak.width = np.clip(peak.width, self.w_min, self.w_max)
                
                    # Append the updated peaks at any given shift-time.
                    self.pre_peaks[run][shift]['peaks'].append(copy.deepcopy(peak))

                temp_peaks = shifted_peaks
        
        # Print for debug
        # for shift in range(self.n_shifts):
        #     print(f"\nShift {shift}.")
        #     for i, peaks in enumerate(self.pre_peaks[shift]['peaks']):
        #         print(f"\tPeak {i+1}. Position: {peaks.position}")
        
    def apply_shift_peaks(self, curr_gen):
        
        # Get current shift
        curr_shift = int(curr_gen / self.shift_interval) 
        
        # Update peaks
        self.peaks = copy.deepcopy(self.pre_peaks[self.args.current_run][curr_shift]['peaks'])
        
        # Update global
        self.update_global_optimum()            

    def shift_peaks(self):
        for peak in self.peaks:
            
            # Flip a random number of bits to shift the peak
            num_bits_to_flip = np.random.randint(1, int(0.3 * self.n) + 1)  # Flip up to N% of bits. TODO: Adjust as hyperparameter for difficulty
            flip_indices = np.random.choice(self.n, num_bits_to_flip, replace=False)
            peak.position[flip_indices] = 1 - peak.position[flip_indices]
            
            # Adjust height and width
            peak.height += np.random.normal(0, 2.5)
            peak.height = np.clip(peak.height, self.h_min, self.h_max)
            peak.width += np.random.normal(0, 0.5)
            peak.width = np.clip(peak.width, self.w_min, self.w_max)
            
        # print(f"Shifted Peaks:")
        # for idx, peak in enumerate(self.peaks):
        #     print(f"\nPeak {idx}. Position: {peak.position}.")
        #     print(f"Height: {peak.height}. Width: {peak.width}")
            
        self.update_global_optimum()

    def update_global_optimum(self):
        """
            Definition
            -----------
                The global optimum is defined as the highest peak.
        """
        if not self.peaks:
            self.global_optimum = None
            return
        self.global_optimum = max(self.peaks, key=lambda peak: peak.height)

    def get_fitness(self, genome):
        """
            Definition
            -----------
                Returns the fitness of the given individual as the highest fitness with respect to all peaks.
        """
        # Fitness is the maximum value among all peaks
        max_fitness = float('-inf')
        
        for peak in self.peaks:
            distance = np.sum(genome != peak.position)  # Hamming distance
            # Gaussian-like function using Hamming distance
            fitness = peak.height * np.exp(- (distance ** 2) / (2 * (peak.width ** 2)))
            if fitness > max_fitness:
                max_fitness = fitness
                
        return max_fitness


