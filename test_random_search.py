from tqdm.auto import trange

from evolve import *

def random_search(T):
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
        new_genome = generate_random_genome(genome_length)
        new_dist = helper(new_genome, spring_indexes, spring_lengths)
        population.append([new_dist, new_genome])
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

population, best_dist_list, best_genome_list = random_search(num_gens)

print(population, best_dist_list, best_genome_list)
