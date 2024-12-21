#knapsack data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import xml.etree.ElementTree as ET
import heapq
import math
from timeit import default_timer as timer
import seaborn as sns
import requests
import textdistance as td
import copy
import scipy.stats as stats

def get_problem(number: int):
    """
    Load knapsack problems
    """
    capacity_url = f"https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{number}_c.txt"
    weights_url = f"https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{number}_w.txt"
    profits_url = f"https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{number}_p.txt"
    solution_url = f"https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{number}_s.txt"

    response_capacity = requests.get(capacity_url)
    capacity = int(response_capacity.text.splitlines()[0].strip())
    
    response_weights = requests.get(weights_url)
    weights = [int(weight.strip()) for weight in response_weights.text.splitlines()]

    response_profits = requests.get(profits_url)
    profits = [int(profit.strip()) for profit in response_profits.text.splitlines()]
    
    response_solution = requests.get(solution_url)
    solution = [int(item.strip()) for item in response_solution.text.splitlines()]

    return capacity, weights, profits, solution

def plot_convergence(df: pd.DataFrame, title: str):
    """
    Plot convergence for repeated runs given a DataFrame with the following columns: generation, fitness, run
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='generation', y='fitness', hue='run')
    plt.title(title)
    plt.xlabel("Generation")
    plt.xlim(0,200)
    plt.ylabel("Fitness")
    plt.show()