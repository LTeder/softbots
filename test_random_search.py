import torch
import numpy as np
from tqdm.auto import tqdm, trange

from bots import Mass, Spring, Universe
from test import *

def random_search(T):
    # generate starting populations
    genome_length = len(spring_lengths)
    population = []
    new_genome = generate_random_genome(genome_length)
    new_dist = helper(new_genome, spring_indexes, spring_lengths)
    population.append([new_dist, new_genome])

    # sorted descending
    population.sort(key = lambda x: x[0], reverse = False)

    size = 1 # starting population size
    best_dist_list = []
    best_genome_list = []
    dist_list = []
    consec_count = 0
    
    for t in trange(T):
        new_genome = generate_random_genome(genome_length)
        new_dist = helper(new_genome, spring_indexes, spring_lengths)
        population.append([new_dist, new_genome])
        population.sort(key = lambda x: x[0], reverse = False)

        population = population[:2]
        
        ### update best dists, tours
        dist, genome = population[0]
        if t == 0:
            best_dist = dist
            best_genome = genome
        elif dist > best_dist:
            best_genome = genome
            best_dist = dist
        
        best_dist_list.append(best_dist)
        best_genome_list.append(best_genome)
        print(f"Population:\n{population}")
        
    return population, best_dist_list, best_genome_list

num_gens = 50

population, best_dist_list, best_genome_list = random_search(num_gens)

print(population, best_dist_list, best_genome_list)
