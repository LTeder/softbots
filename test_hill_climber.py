import numpy as np
from tqdm.auto import trange

from evolve import *

def hill_climber(T, mutat_prob = 1.0):
    # generate starting populations
    genome_length = len(spring_lengths)
    population = []
    new_genome = generate_random_genome(genome_length)
    new_dist = helper(new_genome, spring_indexes, spring_lengths)
    population.append([new_dist, new_genome])

    size = 1 # starting population size
    best_dist_list = []
    best_genome_list = []
    best_dist = 0.
    best_genome = new_genome
    
    for t in trange(T):
        ### Mutation Probability HILL CLIMBING ###
        for _ in range(int(mutat_prob * size)):
            mut_idx = np.random.randint(0, len(population))
            _, genome = population.pop(mut_idx)
            new_genome = mutate_genome(genome)
            new_dist = helper(genome, spring_indexes, spring_lengths)
            population.append([new_dist, new_genome])
            
        ### RECOMBINATION ###
        # generate N offspring
        new_population = []
        idx = np.random.randint(0, len(population))
        [_, parent_1], [_, parent_2] = population[0], population[idx]

        ## keep both offspring
        offspring_1, offspring_2 = crossover(parent_1, parent_2)

        offspring_1_dist = helper(offspring_1, spring_indexes, spring_lengths)
        offspring_2_dist = helper(offspring_2, spring_indexes, spring_lengths)

        new_population.append([offspring_1_dist, offspring_1]) # stays sorted
        new_population.append([offspring_2_dist, offspring_2])

        population += new_population
        population.sort(key = lambda x: x[0], reverse = True)
        
        population = population[:2]
        
        ### update best dists, tours
        dist, genome = population[0]
        if dist > best_dist:
            best_genome = genome
            best_dist = dist
        
        best_dist_list.append(best_dist)
        best_genome_list.append(best_genome)
        print(f"Population:\n{population}")
        
    return population, best_dist_list, best_genome_list

num_gens = 50

population, best_dist_list, best_genome_list = hill_climber(num_gens)

print(population, best_dist_list, best_genome_list)
