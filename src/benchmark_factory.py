import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import math
import numpy as np

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
        self.position = np.array(position)
        self.height = height
        self.width = width
        
class MovingPeaksLandscape:
    """
        Due to its constant shift of landscape it evaluates the resilience of an algorihtm to changes and disruptions
    
    """
    def __init__(self, args, m=5, h_min=1.0, h_max=5.0, w_min=1, w_max=5, shift_interval=30):
        
        # Landscape attributes
        self.n = args.N_NKlandscape           # Genome length
        self.m = m                            # Number of peaks or optima present at any given time.
        self.shift_interval = shift_interval  # Generations between shifts
        self.global_optimum = None            # To store current global optimum
        
        # Peak Height min/max
        self.h_min = h_min
        self.h_max = h_max
        
        # Peak Width min/max
        self.w_min = w_min
        self.w_max = w_max
        
        self.peaks = []
        self.initialize_peaks()
        args.bench_name = 'MovingPeaksLandscape'

    def initialize_peaks(self):
        """
            Definition
            -----------
                Initialize peaks with random positions, heights, and widths.
                    - position: The central point in the search space where the peak is located.
                    - height: The maximum fitness value of the peak.
                    - width: Determines how quickly the fitness value decreases as one moves away from the peak center
        """
        # Initialize peak positions, heights, widths
        self.peaks = []
        for _ in range(self.m):
            position = np.random.randint(0, self.n)
            height = np.random.uniform(self.h_min, self.h_max)
            width = np.random.randint(self.w_min, self.w_max + 1)
            # self.peaks.append({'position': position, 'height': height, 'width': width})
            self.peaks.append(Peak(position, height, width))
            
        self.update_global_optimum()

    def shift_peaks(self):
        """
            At every shift_interval generations, the landscape undergoes a shift where:

                Peak Positions: Move to new locations within the search space.
                Peak Heights: Adjusted randomly within the defined range.
                Peak Widths: Changed to alter the landscape's ruggedness.
        """
        
        # Randomly shift peak positions, heights, and widths
        for peak in self.peaks:
            
            # Shift position
            shift = np.random.randint(-5, 6)
            peak.position = (peak.position + shift) % self.n 
            
            # Adjust height
            peak.height += np.random.normal(0, 0.5)  # Add Gaussian noise
            peak.height = np.clip(peak.height, self.h_min, self.h_max)
            
            # Adjust width
            peak.width += np.random.randint(-1, 2)  # Change width by -1, 0, or +1
            peak.width = np.clip(peak.width, self.w_min, self.w_max)  # Maintain within bounds
    
        self.update_global_optimum()
            
    def update_global_optimum(self):
        """
            Update the current global optimum based on the highest peak.
        """
        if not self.peaks:
            self.global_optimum = None
            return
        self.global_optimum = max(self.peaks, key=lambda peak: peak.height)
        
    def get_current_global_optimum_fitness(self):
        """
            Return the fitness of the current global optimum.
        """
        if self.global_optimum is None:
            return None
        return self.global_optimum.height

    def get_fitness(self, genome):
        """
            To identify the current lobal optimum we need to determine which peak has the highest fitness.
        """
        fitness = 0.0
        for peak in self.peaks:
            distance = min(abs(peak.position - np.sum(genome)), self.n - abs(peak.position - np.sum(genome)))
            fitness += peak.height * np.exp(- (distance ** 2) / (2 * peak.width ** 2))
        return fitness / self.m

