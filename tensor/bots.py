import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Setup execution device for torch tensors
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def dist(p1, p2):
    # p1 and p2 are torch tensors
    return (p2 - p1).pow(2).sum().pow(0.5).item()

def normalize(p):
    # p is a torch tensor
    magnitude = p.pow(2).sum().pow(0.5).item()
    return p / magnitude if magnitude else 0
    
class Mass:
    def __init__(self, p = [0, 0, 0], v = [0, 0, 0], a = [0, 0, 0], F = [0, 0, 0], m = 1):
        self.m = m
        self.p = torch.tensor(p, dtype = torch.float64, device = device) # Position
        self.v = torch.tensor(v, dtype = torch.float64, device = device) # Velocity
        self.a = torch.tensor(a, dtype = torch.float64, device = device) # Acceleration
        self.F = torch.tensor(F, dtype = torch.float64, device = device) # Force

class Spring:
    def __init__(self, Masses, L_0 = 1, k = 1, m1_idx = 0, m2_idx = 0, status = 'steady',
                 damping = 0.0, b = 0, omega = 0, c = 0):
        """
        m1: index in associated Mass list
        m2: index in associated Mass list
        
        omega: frequency of breathing
        c: coefficient of multiplication for breathing
        
        status: element in ['steady', 'compressed', 'stretched'] --> used for plotting later
        """
        self.L_0 = L_0
        self.k = k
        self.m1 = Masses[m1_idx]
        self.m2 = Masses[m2_idx]
        self.status = status
        self.damping = damping
        self.b = b
        self.omega = omega
        self.c = c        
        self.set_L_1()
        
    def set_L_1(self, t = 0):
        self.refresh_L()
        self.L_1 = self.L_0 * (1 + self.b * np.sin(2 * np.pi * self.omega * t + self.c))
          
    def refresh_L(self):
        self.L = dist(self.m1.p, self.m2.p)
        
    def force(self):
        """
        if L > L_1, stretched; apply contraction force (positive)
        if L < L_1, compressed; apply expanding force (negative)
        
        returns force magnitude (direction to be determined outside)
        """
        return self.k * (self.L - self.L_1) * (1 - self.damping)
    
    def energy(self):
        return 1/2 * self.k * (self.L - self.L_1)**2

class Universe:
    def __init__(self, Masses, Springs, dt, box_dims = [20, 20, 20], K_G = 1e5,
                 grav = -9.812, damping = 0, mu = 1.0, verbose = False):
        self.Masses = Masses
        self.Springs = Springs
        self.dt = dt

        self.box_dims = box_dims # bounds
        self.K_G = K_G # resistance for walls
        self.grav = grav # gravitational constant
        self.damping = damping
        self.mu = mu # coefficient of friction
        self.verbose = verbose

        for s in self.Springs:
            s.damping = self.damping
        self.kinetic = 0
        self.potential_springs = 0 # includes bounds
        self.potential_gravity = 0
        self.points = None
        self.energies = None
    
    def display_frame(self, save = False, filename = None):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.cla()
        ax.axes.set_xlim3d(left = 0, right = self.box_dims[0]) 
        ax.axes.set_ylim3d(bottom = 0, top = self.box_dims[1]) 
        ax.axes.set_zlim3d(bottom = 0, top = self.box_dims[2]) 
        
        # scatter plot of Point Masses
        for m in self.Masses:
            ax.scatter3D(*m.p.cpu(), color = 'k')
        
        # plot of Springs
        for s in self.Springs:
            p1 = s.m1.p.cpu()
            p2 = s.m2.p.cpu()
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            z = [p1[2], p2[2]]
            if s.status == 'steady':
                color = 'green'
            elif s.status == 'stretched':
                color = 'red'
            else:
                color = 'blue'
            ax.plot3D(x, y, z, color = color)

        fig.set_size_inches(10, 10)

        if save:
            fig.savefig(filename)

        return ax

    def center_of_mass_horizontal(self):
        com = torch.zeros((2), device = device)
        total_mass = 0
        for m in self.Masses:
            total_mass += m.m
            com += m.p[:2] * m.m
        com /= total_mass
        return com

    def integration_step(self):
        # velocity and position carry over, force and acceleraton are recalculated at each time step
        # reset forces and energies
        for m in self.Masses:
            m.F *= 0
        self.potential_springs = 0
        self.potential_gravity = 0
        self.kinetic = 0

        ### calculate spring forces
        for s in self.Springs:
            magnitude = s.force()
            # update status
            if magnitude == 0:
                s.status = 'steady'
            elif magnitude > 0:
                s.status = 'stretched'
            else:
                s.status = 'compressed'
            # update force tensor attributes
            s.m1.F += normalize(s.m2.p - s.m1.p) * magnitude
            s.m2.F += normalize(s.m1.p - s.m2.p) * magnitude
            # add spring potential energy
            self.potential_springs += s.energy()

        ### update Mass Forces
        for m in self.Masses:
            if self.grav:
                m.F[2] += self.grav * m.m
                # calculate gravitational potential energy
                self.potential_gravity -= self.grav * m.m * m.p[2].item() # based on position 
                # calculate kinetic energy
                self.kinetic += 1/2 * m.m * m.v.pow(2).sum().pow(0.5).item() ** 2
            # ground interaction
            if m.p[2] < 0:
                normal_force = self.K_G * m.p[2].item()
                # calculate elastic energy within ground
                self.potential_springs += 1/2 * self.K_G * m.p[2].item() ** 2
                ## calculate friction force
                #m.F -= normalize(m.v) * self.mu * normal_force
                m.F[:2] -= normalize(m.v[:2]) * self.mu * normal_force
                m.F[2] -= normal_force

        ### update a, v, p
        for m in self.Masses:
            m.a = m.F / m.m # acceleration
            m.v += m.a * self.dt # velocity
            m.p += m.v * self.dt # position

        ### update spring lengths
        for s in self.Springs:
            s.set_L_1(self._t) # _t is updated before integration_step is called

        if self.verbose:
            for i, m in enumerate(self.Masses):
                print(f"### Mass {i} ###\n"
                      f"m.F = {m.F}, m.a = {m.a}, m.v = {m.v}, m.p = {m.p}")
    
    def simulate(self, t):
        """
        t: iterable of time points, spaced by dt
        
        returns the robot mass points, its energies, and its total distance traveled
        """
        digit_length = len(str(len(t)))
        frames = []
        start_pos_horizontal = self.center_of_mass_horizontal()
        
        self.energies = torch.zeros((len(t), 3), device = device)
        self.points = torch.zeros((len(t), len(self.Masses), 3), device = device)

        if self.verbose:
            for i, m in enumerate(self.Masses):
                print(f"### Mass {i} ###\n"
                      f"m.F = {m.F}, m.a = {m.a}, m.v = {m.v}, m.p = {m.p}")
        
        for i, _t in tqdm(enumerate(t), total = len(t)):
            self._t = _t # updated for Spring.set_L_1() called next
            self.integration_step() # updates energies and self.Masses
            self.energies[i] = torch.tensor([self.kinetic, self.potential_springs,
                                             self.potential_gravity], device = device)
            # update points iteratively with Masses
            for j, m in enumerate(self.Masses):
                self.points[i][j] = m.p.clone()

        end_pos_horizontal = self.center_of_mass_horizontal()
        total_dist_horizontal = dist(start_pos_horizontal, end_pos_horizontal)
            
        return self.points, self.energies, total_dist_horizontal
