
import pandas as pd
import numpy as np
import textdistance as td
from ga import GA
from sequence_tracker import SequenceTracker
from simulated_annealing import run_simulated_annealing

class CrisprGA(GA):
    def __init__(self, population_size,
                generations, profits, weights, capacity,
                threshold=0.5, items=10, mating_pool_size=5, 
                population=None, apply_mutation_probability=.8, apply_crispr_probability=.8, learning_phase=10,
                sequence_length=2, fittest_individuals=4, metric='jaro_winkler', dna_repair=True):
        super().__init__(population_size, generations, profits, 
                        weights, capacity, items=items, 
                        mating_pool_size=mating_pool_size, population=population)
        self.sequence_tracker = SequenceTracker()
        self.learning_phase = learning_phase
        self.apply_mutation_probability = apply_mutation_probability
        self.apply_crispr_probability = apply_crispr_probability
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.metric = metric
        self.dna_repair = dna_repair
        self.fittest_individuals = fittest_individuals
        self.string_metrics = {
            'jaro': td.jaro.normalized_similarity,
            'jaro_winkler': td.jaro_winkler.normalized_similarity,
            'levenshtein': td.levenshtein.normalized_similarity,
            'cosine': td.cosine.normalized_similarity,
            'hamming': td.hamming.normalized_similarity
        }
    
    def get_fittest_individuals(self, population: pd.DataFrame, n=2):
        """
        Returns the n fittest individuals stored in the population DataFrame.
        """
        df = population.copy()
        df.sort_values(by='fitness', ascending=False, inplace=True)
        return df['solution'].head(n).values

    def string_similarity(self, solutions: np.array, metric, sequence_length: int, generation: int, threshold: float):
        """
        Identify similar sequences with a sliding window in a group of solutions.
        """

        for i in range(len(solutions)):
            s1 = list(map(str, solutions[i]))
            for j in range(i+1, len(solutions)):
                s2 = list(map(str, solutions[j]))
                for k in range(len(s1) - (sequence_length + 1)):
                    subsequence_1 = s1[k:k+sequence_length]
                    subsequence_2 = s2[k:k+sequence_length]
                    if metric(subsequence_1, subsequence_2) > threshold:
                        self.sequence_tracker.add_sequence(k, subsequence_1, generation)
    
    def guided_insertion(self, solution, n_recent_generations: int, start: int):
        """
        Performs guided insertion by selecting a sequence from the sequence tracker (BB Array) and inserting it into the solution.
        The sequence is selected based on the frequency of occurrence and filtered by the most recent generations.
        """
        sequence = self.sequence_tracker.select_sequence(start, n_recent_generations)
        if sequence is not None:
            solution[start:start+len(sequence)] = sequence
        return solution

    
    def get_substring(self, string, position, sequence_length=3):
        """
        Get a subsequence of a given length from a starting position.
        """
        return string[position:position+sequence_length]


    def dna_repairs(self, solution, previous_fitness, max_iter=10):
        """
        Perform a local search repair on the solution to improve the fitness of the solution by hill climbing.
        This is performed to repair mutations that resulted in a decrease in fitness.
        """
        current_fitness = self.get_fitness(solution)

        if current_fitness >= previous_fitness:
            return solution
        self.ran_simulated_annealing = 1
        candidate_solution = run_simulated_annealing(solution, self.weights, self.profits, self.capacity, 1000, 0.99, max_iter)
        return candidate_solution

    def run(self):
        """
        Run the CRISPR GA algorithm to solve the knapsack problem for a given number of generations.
        """
        # track fitness results 
        fitness_results = []
        population_df = pd.DataFrame({"generation": 0, "solution": self.population, "fitness": self.get_population_fitness(self.population)})
        for generation in range(self.generations):
            # perform selection
            mating_pool = self.roulette_wheel_selection(population_df['solution'])

            # random selection of parents from mating pool
            parents = mating_pool[np.random.choice(len(mating_pool), 2, replace=False)]
            parent_1, parent_2 = parents

            fittest_individuals = self.get_fittest_individuals(population_df, self.fittest_individuals)
            self.string_similarity(fittest_individuals, self.string_metrics[self.metric], self.sequence_length, generation, self.threshold)
            # crossover
            offspring_1, offspring_2 = self.crossover(parent_1, parent_2)

            # perform mutation on every generation in the learning phase
            if generation < self.learning_phase:
                offspring_1 = self.mutation(offspring_1)
                offspring_2 = self.mutation(offspring_2)
                
            if generation > self.learning_phase:

                # perform mutation with a probability
                if np.random.rand() < self.apply_mutation_probability:
                    offspring_1 = self.mutation(offspring_1)
                    offspring_2 = self.mutation(offspring_2)

                # perform crispr with a probability
                # this will be updated to use a per allele insertion by flipping a coin for each position
                if np.random.rand() < self.apply_crispr_probability:
                    previous_fitness_1 = self.get_fitness(offspring_1.copy())
                    previous_fitness_2 = self.get_fitness(offspring_2.copy())

                    offspring_1 = self.guided_insertion(offspring_1, generation - 10, np.random.randint(self.items - self.sequence_length))
                    offspring_2 = self.guided_insertion(offspring_2, generation - 10, np.random.randint(self.items - self.sequence_length))

                    if self.dna_repair:
                        offspring_1 = self.dna_repairs(offspring_1, previous_fitness_1)
                        offspring_2 = self.dna_repairs(offspring_2, previous_fitness_2)
                    
                    offspring_1 = self.repair_solution(offspring_1)
                    offspring_2 = self.repair_solution(offspring_2)

            # update the population, sort by the lowest fitness
            population_df = population_df.sort_values(by='fitness', ascending=True)
            population_df = population_df.iloc[2:]

            offspring_df = pd.DataFrame({"generation": generation, "solution": [offspring_1, offspring_2], "fitness": [self.get_fitness(offspring_1), self.get_fitness(offspring_2)]})
            population_df = pd.concat([offspring_df, population_df], ignore_index=True)
            fitness_results.append({"generation": generation, "fitness": population_df['fitness'].max(), "algorithm": 'crispr'})
        return population_df, pd.DataFrame(fitness_results)
    
if __name__ == "__main__":
    pass