
import numpy as np
import pandas as pd
from ga import GA

# EDA is an Evolutionary Algorithm, but it can use some of the same methods as the GA class. The population is evolved with a steady state approach.
class EDA(GA):
    def __init__(self, population_size, 
                 generations, profits, weights, capacity,
                 items=10, mating_pool_size=5, population=None, learning_rate=0.5, beta=0.5):
        super().__init__(population_size, generations, profits, 
                         weights, capacity, items=items, 
                         mating_pool_size=mating_pool_size, population=population)
        self.learing_rate = learning_rate
        self.beta = beta

    def generate_probability_vector(self, solutions):
        """
        Generate a probability vector for a binary string population based on the value at each index.
        """
        n = len(solutions)
        # the probability of each index being 1
        probabilities = [sum([solution[i] for solution in solutions]) / n for i in range(len(solutions[0]))]
        return np.array(probabilities)

    def update_probability_vector(self, solutions, probability_vector, learning_rate):
        """
        Update the probability vector based on the current solutions in the population.
        """
        temp_vector = self.generate_probability_vector(solutions)

        probability_vector = (1 - learning_rate) * probability_vector + (learning_rate * temp_vector)
        return probability_vector
        

    def guided_mutation(self, solution, probability_vector, beta):
        """
        Perform a guided mutation on the solution based on the probability vector.
        """
        n = len(solution)
        for i in range(n):
            # flip a coin with probability beta
            if np.random.rand() < beta:
                # flip the bit at index i based on the probability vector at index i
                try:
                    if np.random.rand() < probability_vector[i]:
                        solution[i] = 1
                    else:
                        solution[i] = 0
                except:
                    print(i)
                    print(probability_vector)
                    print(solution)
            # otherwise, keep the solution the same
        return solution
    
    def run(self):
        """
        Run the EDA algorithm for a set number of generations.
        """
        fitness_results = []
        population_df = pd.DataFrame({"generation": 0, "solution": self.population, "fitness": self.get_population_fitness(self.population)})
        probability_vector = None
        for generation in range(self.generations):
            # perform selection
            mating_pool = self.roulette_wheel_selection(population_df['solution'])
            # print("performed mating pool selection")
            # random selection of parents from mating pool
            parents = mating_pool[np.random.choice(len(mating_pool), 2, replace=False)]
            parent_1, parent_2 = parents
            
            if generation == 0:
                    probability_vector = self.generate_probability_vector([parent_1, parent_2])
            probability_vector = self.update_probability_vector([parent_1, parent_2], probability_vector, self.learing_rate)
            offspring_1 = self.guided_mutation(parent_1, probability_vector, self.beta)
            offspring_2 = self.guided_mutation(parent_2, probability_vector, self.beta)
            offspring_1 = self.repair_solution(offspring_1)
            offspring_2 = self.repair_solution(offspring_2)

            # update the population, sort by the lowest fitness
            population_df = population_df.sort_values(by='fitness', ascending=True)
            population_df = population_df.iloc[2:]

            offspring_df = pd.DataFrame({"generation": generation, "solution": [offspring_1, offspring_2], "fitness": [self.get_fitness(offspring_1), self.get_fitness(offspring_2)]})
            population_df = pd.concat([offspring_df, population_df], ignore_index=True)

            fitness_results.append({"generation": generation, "fitness": population_df['fitness'].max(), "algorithm": 'eda'})

        return population_df, pd.DataFrame(fitness_results)

if __name__ == "__main__":
    pass