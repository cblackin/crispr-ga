
import copy
import math
import numpy as np

def linear_cooling(cool_rate: int, temperature: int):
    """
    Linear cooling schedule
    """
    current_temperature = temperature
    current_temperature *= cool_rate
    return current_temperature

def get_fitness(solution, weights, profits, capacity):
    """
    Fitness function for the knapsack problem (maximize profit while not exceeding capacity)
    """
    weight = sum(included * weight for included, weight in zip(solution, weights))
    profit = sum(included * profit for included, profit in zip(solution, profits)) 

    # not feasible
    if weight > capacity:
        return 0
    return profit


def get_neighborhood(solution):
    """
    Generate a neighbor solution by flipping a random bit in the candidate solution
    """
    neighbor = copy.deepcopy(solution)
    index = np.random.randint(4, size=1)
    index = index[0]
    # remove or exclude the item in the candidate solution
    try:
        neighbor[index] = 1 - neighbor[index]
    except:
        print(index)
        print(neighbor)
    return neighbor

def boltzmann_probability(delta_energy, temperature):
    """
    Probability of accepting a solution based on the Boltzmann probability
    The probability of accepting a solution will be high at higher temperatures, and low at lower temperatures.
    Therefore, the algorithm will explore global solutions at the beginning, then exploit local best tours at the end.
    """
    return math.exp(-(abs(delta_energy)/temperature))

def run_simulated_annealing(solution, weights, profits, capacity, 
                                temperature, cool_rate, max_iter):
    """
    Run the simulated annealing algorithm to solve the knapsack problem for a given number of iterations.
    """
    candidate_solution = copy.deepcopy(solution)
    candidate_fitness = get_fitness(solution, weights, profits, capacity)

    current_temperature = temperature

    for i in range(1, max_iter):
        if current_temperature > 0:
            neighbor = get_neighborhood(candidate_solution)
            neighbor_fitness = get_fitness(neighbor, weights, profits, capacity)
            if neighbor_fitness > 0:
                delta_energy = neighbor_fitness - candidate_fitness

                if delta_energy > 0 or np.random.random() < boltzmann_probability(delta_energy, current_temperature):
                    candidate_solution = copy.deepcopy(neighbor)
                    candidate_fitness = copy.deepcopy(neighbor_fitness)

                current_temperature = linear_cooling(cool_rate, current_temperature)

    return candidate_solution

if __name__ == "__main__":
    pass