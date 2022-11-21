import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from bots import Spring, Mass, Universe

# Setup execution device for torch tensors
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
device = torch.device("cpu")
device

class Soft_Spring(Spring):
    def __init__(self, Masses, L_0=1, k=1000, m1_idx=0, m2_idx=0,
                 status='steady', damping=0.0):
        super().__init__(Masses, L_0=L_0, k=k, m1_idx=m1_idx, m2_idx=m2_idx,
                         status=status, damping=damping, b=0, omega=0, c=0)

class Hard_Spring(Spring):
    def __init__(self, Masses, L_0=1, k=20000, m1_idx=0, m2_idx=0,
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


Masses = [
    ## upper layer
        # left side (0, 2, 4, 6)            # right side (1, 3, 5, 7)
         Mass(m=1, p=[9, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 3], v=[0,0,0]),
         Mass(m=1, p=[6, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 3], v=[0,0,0]),
         Mass(m=1, p=[3, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 3], v=[0,0,0]),
         Mass(m=1, p=[0, 2, 3], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 3], v=[0,0,0]),

    ## lower layer
        # left side (8, 10, 12, 14)          # right side (9, 11, 13, 15)
         Mass(m=1, p=[9, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[9, 5, 0], v=[0,0,0]),
         Mass(m=1, p=[6, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[6, 5, 0], v=[0,0,0]),
         Mass(m=1, p=[3, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[3, 5, 0], v=[0,0,0]),
         Mass(m=1, p=[0, 2, 0], v=[0,0,0]), Mass(m=0.1, p=[0, 5, 0], v=[0,0,0])]

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
uni = Universe(Masses, Springs, dt, verbose=False)

t = np.arange(0, 2.0, dt)
result = uni.simulate(t)

points_tensor = result[0].clone().detach()

torch.save(points_tensor, 'Robot4_dt0.0001_damping0.25.pt')
