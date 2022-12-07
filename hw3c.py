from cppn_encoding import *

def main():
    ### GENETIC ALGORITHM
    # (depth, N, pop_size, num_gens, T, dt = 0.0001, p = 0.5, mutat_prob = 0.05, damping=0.05, 
    #                         constant_max = 1):
    depth = 6
    N = 3
    pop_size = 20
    num_gens = 100
    T = 5
    dt = 0.0001
    truncation_p = 0.5
    mutation_prob = 0.25
    damping = 0.05

    results = genetic_programming(depth, N, pop_size, num_gens, T, dt = dt,
                                  p = truncation_p, mutat_prob = mutation_prob,
                                  damping = damping)
    population, best_dist_list, best_genome_list, diversity_list = results
    
    print(f'best dist list:{best_dist_list}\n')
    print(f'best genome list:{best_genome_list}\n')
    print(f'diversity list:{diversity_list}\n')
    
    
if __name__ == '__main__':
    main()
