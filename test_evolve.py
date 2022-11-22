import torch
import numpy as np
from tqdm.auto import tqdm, trange

from evolve import *

## simple: only crossover
def genetic_programming(N, T, p = 0.5, mutat_prob = 0.05):
    # generate starting populations
    genome_length = len(spring_lengths)
    population = []
    print("Generating initial population...")
    for _ in trange(N): # Generate initial population
        new_genome = generate_random_genome(genome_length)
        new_dist = helper(new_genome, spring_indexes, spring_lengths)
        population.append([new_dist, new_genome])

    # sorted descending
    population.sort(key = lambda x: x[0], reverse=False)

    size = N # starting population size
    best_dist_list = []
    best_genome_list = []
    dist_list = []
    consec_count = 0
    
    print("Beginning evolution...")
    for t in trange(T):
        ### Mutation Probability HILL CLIMBING ###
        for mutation in range(int(mutat_prob*size)):
            mut_idx = np.random.randint(0, len(population))
            _, genome = population.pop(mut_idx)
            new_genome = mutate_genome(genome)
            new_dist = helper(genome, spring_indexes, spring_lengths)
            population.append([new_dist, new_genome])

        population.sort(key=lambda x: x[0], reverse=False)
            
        ### RECOMBINATION ###
        new_population = []
        # to-do: implement niching
        idx_1, idx_2 = np.random.choice(len(population), 2, replace=False)
        [_, parent_1], [_, parent_2] = population[idx_1], population[idx_2]

        ## keep both offspring
        offspring_1, offspring_2 = crossover(parent_1, parent_2)

        offspring_1_dist = helper(offspring_1, spring_indexes, spring_lengths)
        offspring_2_dist = helper(offspring_2, spring_indexes, spring_lengths)

        new_population.append([offspring_1_dist, offspring_1]) # stays sorted
        new_population.append([offspring_2_dist, offspring_2])

        population += new_population
        population.sort(key = lambda x: x[0], reverse = False)
        
        population = population[:1]
        
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
pop_size = 10

population, best_dist_list, best_genome_list = \
    genetic_programming(pop_size, num_gens, p = 0.5, mutat_prob = 0.05)

print(population, best_dist_list, best_genome_list)
