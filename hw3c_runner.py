from cppn_encoding import *

def main():
    ### GENETIC ALGORITHM

    # (depth, N, pop_size, num_gens, T, dt = 0.0001, p = 0.5, mutat_prob = 0.05, damping=0.05, 
    #                         constant_max = 1):
    depth = 6
    N = 3
    pop_size = 10
    num_gens = 50
    T = 5
    dt = 0.0001
    truncation_p = 0.5
    mutation_prob = 0.25
    damping = 0.05


    population, best_dist_list, best_genome_list, diversity_list = genetic_programming(depth, N,
                                                                       pop_size, num_gens, 
                                                                       T, dt = dt,
                                                                       p = truncation_p, mutat_prob = mutation_prob,
                                                                       damping = damping)
    print('best dist list:')
    print(best_dist_list)
    print()
    
    print('best genome list:')
    print(best_genome_list)
    print()
    
    print('diversity list:')
    print(diversity_list)
    print()
    
    
if __name__ == '__main__':
    main()