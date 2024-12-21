import numpy as np
import copy
import pandas as pd
from ga import GA

class GOMEA(GA):
    def __init__(self, population_size, 
                 generations, profits, weights, capacity, linkage_model: list,
                 mutation_rate=0.01, items=10, mating_pool_size=5,
                 population=None):
        super().__init__(population_size, generations, profits, 
                         weights, capacity, mutation_rate=mutation_rate, items=items, 
                            mating_pool_size=mating_pool_size, population=population)
        self.linkage_model = linkage_model
    
    def optimal_mixing(self, solution, population, linkage_model):
        """
        The optimal mixing operator for the GOMEA algorithm. For each group in the linkage model, a random donor is selected from the population and the genes in the group are swapped with the donor.
        If the new candidate has a higher objective fitness, the candidate is updated to the current candidate.
        """
        # copy the solution as a candidate and get the starting fitness
        candidate = copy.deepcopy(solution)
        current_fitness = self.get_fitness(solution)
        # iterate through the linkage model and try swapping the genes with a new donor for each group
        for group in linkage_model:

            current_candidate = copy.deepcopy(candidate)
            index = np.random.choice(len(population), 1, replace=False)[0]
            donor = population[index]

            for gene_index in group:
                current_candidate[gene_index] = donor[gene_index]
            
            # if the new candidate is better, update the candidate to the current candidate
            new_fitness = self.get_fitness(current_candidate)
            if new_fitness > current_fitness:
                current_fitness = new_fitness
                candidate = copy.deepcopy(current_candidate)

        return candidate

    def run(self):
        """
        The GOMEA algorithm for solving the knapsack problem. The algorithm evolves the population by selecting a random parent and performing optimal mixing with a random subset of the population.
        """
        fitness_results = []
        population = self.population
        for generation in range(self.generations):
            
            index = np.random.choice(len(population), 1, replace=False)[0]
            parent = population[index]

            random_subset = np.random.choice(len(self.linkage_model), 10, replace=False)
            population_subset = [population[i] for i in random_subset]
    
            offspring = self.optimal_mixing(parent, population_subset, self.linkage_model)
            # replace the parent with the offspring
            population[index] = offspring

            fitness_results.append({"generation": generation, "fitness": np.max(self.get_population_fitness(population)), "algorithm": 'gomea'})

        return pd.DataFrame(fitness_results)