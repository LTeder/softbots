import numpy as np

from bots import Mass, Spring, Universe

DAMPING = 0.5 # Damping factor within each robot simulation's universe
T = 5 # Seconds of simulation runtime per robot

"""# Setup

## Spring Library
Note: breathing: a * cos(2$\pi$ * b + c)


to evolve: length of a, as well as types of springs

* soft: k=1000, b=c=0
* hard: k=20000, b=c=0
* contract, expand: k=5000, b$\in$[-.5, .5], c = 0
* expand, contract: k=5000, b$\in$[-.5, .5], c = $\pi$
"""

class Soft_Spring(Spring):
    def __init__(self, Masses, L_0=1, k=1000, m1_idx=0, m2_idx=0,
                 status='steady', damping=0.0):
        super().__init__(Masses, L_0=L_0, k=k, m1_idx=m1_idx, m2_idx=m2_idx,
                         status=status, damping=damping, b=0, omega=0, c=0)

class Hard_Spring(Spring):
    def __init__(self, Masses, L_0=1, k=50000, m1_idx=0, m2_idx=0,
                 status='steady', damping=0.0):
        super().__init__(Masses, L_0=L_0, k=k, m1_idx=m1_idx, m2_idx=m2_idx,
                         status=status, damping=damping, b=0, omega = 0, c=0)

class ExpandContract(Spring):
    def __init__(self, Masses, L_0=1, k=5000, m1_idx=0, m2_idx=0,
                 status='steady', damping=0.0, b=0.1, omega=0.5):
        super().__init__(Masses, L_0=L_0, k=k, m1_idx=m1_idx, m2_idx=m2_idx,
                         status=status, damping=damping, b=b, omega = omega, c=0)

class ContractExpand(Spring):
    def __init__(self, Masses, L_0=1, k=5000, m1_idx=0, m2_idx=0,
                 status='steady', damping=0.0, b=0.1, omega=0.5):
        super().__init__(Masses, L_0=L_0, k=k, m1_idx=m1_idx, m2_idx=m2_idx,
                         status=status, damping=damping, b=b, omega = omega, c=np.pi)

"""## Robot Morphology
Springs description 3 (forward-back asymmetrical): 
* Cross Y springs are soft (8)
* Z springs are soft (8)
* 1st and 2nd cube: diags and X springs are expanding-> contracting (16)
* Last cube: diags and X springs are contracting -> expanding (8)
"""

Masses = [
    ## upper layer
        # left side (0, 2, 4, 6)            # right side (1, 3, 5, 7)
         Mass(m=0.1, p=[9, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 3], v=[0,0,0]),
         Mass(m=0.1, p=[6, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 3], v=[0,0,0]),
         Mass(m=0.1, p=[3, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 3], v=[0,0,0]),
         Mass(m=0.1, p=[0, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 3], v=[0,0,0]),

    ## lower layer
        # left side (8, 10, 12, 14)          # right side (9, 11, 13, 15)
         Mass(m=0.1, p=[9, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 0], v=[0,0,0]),
         Mass(m=0.1, p=[6, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 0], v=[0,0,0]),
         Mass(m=0.1, p=[3, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 0], v=[0,0,0]),
         Mass(m=0.1, p=[0, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 0], v=[0,0,0])]

Springs = [
    ### cross Y springs
    Hard_Spring(Masses, L_0 = 3, m1_idx = 0, m2_idx = 1),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 2, m2_idx = 3),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 4, m2_idx = 5),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 6, m2_idx = 7),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 8, m2_idx = 9),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 10, m2_idx = 11),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 12, m2_idx = 13),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 14, m2_idx = 15),

    ### Z springs
    # left
    Hard_Spring(Masses, L_0 = 3, m1_idx = 0, m2_idx = 8),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 2, m2_idx = 10),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 4, m2_idx = 12),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 6, m2_idx = 14),
    # right
    Hard_Spring(Masses, L_0 = 3, m1_idx = 1, m2_idx = 9),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 3, m2_idx = 11),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 5, m2_idx = 13),
    Hard_Spring(Masses, L_0 = 3, m1_idx = 7, m2_idx = 15),

    ### 1st and 2nd cubes (expanding -> contracting)
    ## 1st cube
    # diagonal
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 0, m2_idx = 11),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 1, m2_idx = 10),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 8, m2_idx = 3),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 9, m2_idx = 2),

    # x horizontal
    ContractExpand(Masses, L_0 = 3, m1_idx = 0, m2_idx = 2),
    ContractExpand(Masses, L_0 = 3, m1_idx = 1, m2_idx = 3),
    ContractExpand(Masses, L_0 = 3, m1_idx = 8, m2_idx = 10),
    ContractExpand(Masses, L_0 = 3, m1_idx = 9, m2_idx = 11),    

    ## 2nd cube
    # diagonal
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 2, m2_idx = 13),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 3, m2_idx = 12),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 10, m2_idx = 5),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 11, m2_idx = 4),

    # x horizontal
    ContractExpand(Masses, L_0 = 3, m1_idx = 2, m2_idx = 4),
    ContractExpand(Masses, L_0 = 3, m1_idx = 3, m2_idx = 5),
    ContractExpand(Masses, L_0 = 3, m1_idx = 10, m2_idx = 12),
    ContractExpand(Masses, L_0 = 3, m1_idx = 11, m2_idx = 13),    

    ### last cube (contracting -> expanding)
    # diagonal
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 4, m2_idx = 15),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 5, m2_idx = 14),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 12, m2_idx = 7),
    ContractExpand(Masses, L_0 = 27**0.5, m1_idx = 13, m2_idx = 6),

    # x horizontal
    ContractExpand(Masses, L_0 = 3, m1_idx = 4, m2_idx = 6),
    ContractExpand(Masses, L_0 = 3, m1_idx = 5, m2_idx = 7),
    ContractExpand(Masses, L_0 = 3, m1_idx = 12, m2_idx = 14),
    ContractExpand(Masses, L_0 = 3, m1_idx = 13, m2_idx = 15)]
        
dt = 0.0001
uni = Universe(Masses, Springs, dt)

"""# Genetic Setup

## Get spring indexes and lengths for genome
"""

spring_indexes = []
spring_lengths = []

for s in Springs:
    spring_indexes.append((s.m1, s.m2))
    spring_lengths.append(s.L_0)

def genome_to_robot(Masses, genome, spring_indexes, spring_lengths):
    """
    genome: a list of lists

    returns a list of springs
    """
    springs = []

    for i, spring_type in enumerate(genome):
        m1, m2 = spring_indexes[i]
        Masses = [m1, m2]
        L_0 = spring_lengths[i]
        if spring_type == 1:
            new_spring = Soft_Spring(Masses, L_0=L_0, m1_idx=0, m2_idx=1)
        elif spring_type == 2:
            new_spring = Hard_Spring(Masses,L_0=L_0, m1_idx=0, m2_idx=1)
        elif spring_type == 3:
            new_spring = ExpandContract(Masses,L_0=L_0, m1_idx=0, m2_idx=1)
        elif spring_type == 4:
            new_spring = ContractExpand(Masses,L_0=L_0, m1_idx=0, m2_idx=1)

        # check if diagonal - if so, then increase spring constant
        if i in [16,17,18,19, 
                 24, 25, 26, 27, 
                 32, 33, 34, 35]:

                 new_spring.k = 50000
                 
        springs.append(new_spring)

    return springs

def generate_random_genome(length):
    genome = []
    for i in range(length):
        genome.append( np.random.randint(1, 5) )
    return genome

"""## Mutating robots

Types of mutations:
- switching out type of Spring (big)
- varying parameters of Springs
    - spring coefficient
        - limit bounds depending on spring type?
    - breathing offset c (?)
        - if applicable ?
    - breathing amplitude b

Things to NOT mutate:
- breathing PHASE (?)
- or at least if you do, keep it within certain bounds
- resting spring length
"""

def mutate_genome(genome):
    # get random index
    rand_index = np.random.randint(len(genome))

    while True:
        rand_gene = np.random.randint(1, 5)
        if rand_gene != genome[rand_index]:
            genome[rand_index] = rand_gene
            break
        
    return genome

"""## Crossing over robots
Procedure:
- pick a random spring, and then switch?
    - what if multiple? does this need to be adjacnet? 
"""

def crossover(genome1, genome2):
    # pick a random index
    idx = np.random.randint(len(genome1))

    # return two new genomes
    temp = genome1[idx]
    genome1[idx] = genome2[idx]
    genome2[idx] = temp

    return genome1, genome2

"""# Genetic programming"""

def helper(genome, spring_indexes, spring_lengths):
    Masses = [
    ## upper layer
        # left side (0, 2, 4, 6)            # right side (1, 3, 5, 7)
            Mass(m=0.1, p=[9, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 3], v=[0,0,0]),
            Mass(m=0.1, p=[6, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 3], v=[0,0,0]),
            Mass(m=0.1, p=[3, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 3], v=[0,0,0]),
            Mass(m=0.1, p=[0, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 3], v=[0,0,0]),

    ## lower layer
        # left side (8, 10, 12, 14)          # right side (9, 11, 13, 15)
            Mass(m=0.1, p=[9, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 0], v=[0,0,0]),
            Mass(m=0.1, p=[6, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 0], v=[0,0,0]),
            Mass(m=0.1, p=[3, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 0], v=[0,0,0]),
            Mass(m=0.1, p=[0, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 0], v=[0,0,0])]

    springs = genome_to_robot(Masses, genome, spring_indexes, spring_lengths)

    universe = Universe(Masses, springs, dt, damping = DAMPING)

    t = np.arange(0, T, dt)
    points, energies, total_dist = universe.simulate(t)
    
    return total_dist
