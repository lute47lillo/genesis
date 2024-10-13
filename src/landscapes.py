"""
    GENESIS
    
    Landscape classes and functions
    
    Author: Lute Lillo
    
    Date: 13/10/2024

"""

import numpy as np

class NKLandscape:
    """ N-K Fitness Landscape """
    def __init__(self, n=10, k=2):
        self.n = n  # The number of genes (loci) in the genome. Genome length
        self.k = k  # Number of other loci interacting with each gene. For each gene 2^(k+1) possible combinations of gene states that affect its fitness contribution
        self.gene_contribution_weight_matrix = np.random.rand(n, 2**(k+1)) 
        
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