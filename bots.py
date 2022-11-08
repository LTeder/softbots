import torch
import numpy as np
import matplotlib.pyplot as plt

class Mass:
    def __init__(self, m=1, p=[0,0,0], v=[0, 0, 0], a=[0, 0, 0], F=[0, 0, 0], grounded=False, damping=0):
        self.m = m
        self.p = p
        self.v = v
        self.a = a
        self.F = F
        self.grounded = grounded
        self.damping = damping

class Spring:
    def __init__(self, Masses, L_0=1, k=1, m1_idx=0, m2_idx=0, status='steady', damping = 0.0, omega = 1, c = 0):
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
        self.omega = omega
        self.c = c
        
        self.L = self.dist(self.m1.p, self.m2.p)
        self.set_L_1()

    def dist(self, p1, p2):
        """
        p1: (x1, y1, z1)
        p2: (x2, y2, z2)
        """
        return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
        
    def set_L_1(self, t = 0):
        self.L_1 = self.L_0 * (1 + self.c * np.sin(2*np.pi * self.omega * t))
          
    def refresh_L(self):
        self.L = self.dist(self.m1.p, self.m2.p)
        
    def force(self):
        """
        if L > L_1, stretched; apply contraction force (positive)
        if L < L_1, compressed; apply expanding force (negative)
        
        returns force magnitude (direction to be determined outside)
        """
        return self.k * (self.L - self.L_1) * (1-self.damping)
    
    def energy(self):
        return 1/2 * self.k * (self.L - self.L_1)**2 

class Universe:
    def __init__(self, Masses, Springs, dt, box_dims = [10, 10, 10],
                 K_G=1e5, g=-9.812, grav=False, damping = 0, mu=1.0):
        self.Masses = Masses
        self.Springs = Springs
        self.dt = dt
        self.box_dims = box_dims # bounds
        self.K_G=K_G # resistance for walls
        self.g = g # gravitational constant
        self.grav = grav
        self.damping = damping
        self.mu = mu # coefficient of friction
        
        self.DIMENSIONS = 3
        
        for s in self.Springs:
            s.damping = self.damping
            
        self.ax = plt.axes(projection='3d')

        self.points = []
        
        self.kinetic = 0
        self.potential_springs = 0
        self.potential_gravity = 0
        self.energies = [] # kinetic, spring potential (including bounds), gravitational potential
            
    def calculate_mass_center(self):
        """
        Loop through all masses and return the center of mass
        """
        pass

    def normalize_2d(self, p):
        x, y = p
        
        if x==y==0:
            return p
        
        mag = (x**2 + y**2)**0.5
        return [x/mag, y/mag]

    def normalize(self, p):
        x, y, z = p
        
        if x==y==z==0:
            return p
        
        mag = (x**2 + y**2 + z**2)**0.5
        return [x/mag, y/mag, z/mag]

    def integration_step(self, t=0, verbose=False):
        # velocity and position carry over, Force and acceleraton are recalculated at each time step
        
        # reset forces and accelerations
        for m in self.Masses:
            m.F = [0, 0, 0]
            m.a = [0, 0, 0]
            
        # reset energies
        self.potential_springs = 0
        self.potential_gravity = 0
        self.kinetic = 0
        
        ### calculate spring forces
        for s in self.Springs:
            magnitude = s.force()
                        
            if magnitude == 0:
                s.status = 'steady'
            elif magnitude > 0:
                s.status = 'stretched'
            else:
                s.status = 'compressed'
            
            # force on m1
            m1_direction = [s.m1.p[0] - s.m2.p[0], s.m1.p[1] - s.m2.p[1]]
            m1_direction = [s.m2.p[0] - s.m1.p[0], s.m2.p[1] - s.m1.p[1], s.m2.p[2] - s.m1.p[2]]
            m1_direction = self.normalize(m1_direction)
            m1_force = [m1_direction[axis] * magnitude for axis in range(self.DIMENSIONS)]
            
            s.m1.F[0] += m1_force[0] # x
            s.m1.F[1] += m1_force[1] # y
            s.m1.F[2] += m1_force[2] # z
            
            # force on m2
#             m2_direction = [s.m2.p[0] - s.m1.p[0], s.m2.p[1] - s.m1.p[1]]
            m2_direction = [s.m1.p[0] - s.m2.p[0], s.m1.p[1] - s.m2.p[1], s.m1.p[2] - s.m2.p[2]]

            m2_direction = self.normalize(m2_direction)
            m2_force = [m2_direction[axis] * magnitude for axis in range(self.DIMENSIONS)]
            
            s.m2.F[0] += m2_force[0] # x
            s.m2.F[1] += m2_force[1] # y
            s.m2.F[2] += m2_force[2] # z
            
            ### add spring potential energy
            self.potential_springs += s.energy()
        
        ### update Mass Forces
        for m in self.Masses:
            
            ### gravity
            if self.grav:
                m.F[2] += self.g * m.m
                
                # calculate gravitational potential energy
                self.potential_gravity += - self.g * m.m * m.p[2] # based on position 
                
                # calculate kinetic energy
                self.kinetic += 1/2 * m.m * (m.v[0]**2 + m.v[1]**2 + m.v[2]**2)
                
            # ground
            if m.p[2] < 0:
                # normal force
                normal_force = self.K_G * (0 - m.p[2])
                m.F[2] += normal_force
                # calculate elastic energy
                self.potential_springs += 1/2 * self.K_G * (0 - m.p[2])**2

                ## calculate friction force
                if not (m.p[0] == 0 and m.p[1] == 0):
                    # calculate direction of movement
                    x_normed, y_normed = self.normalize_2d(m.v[:2]) 
                    # oppose x direction
                    m.F[0] -= x_normed * self.mu * normal_force

                    # oppose y direction
                    m.F[1] -= y_normed * self.mu * normal_force
            
        ### calculate energies (note: should this be before or after the points are adjusted?)
        ### update a, v, p
        for m in self.Masses:
            # update acceleration
            m.a[0] = m.F[0] / m.m # x
            m.a[1] = m.F[1] / m.m # y
            m.a[2] = m.F[2] / m.m # z
            
            # update velocity
            m.v[0] += m.a[0] * self.dt
            m.v[1] += m.a[1] * self.dt
            m.v[2] += m.a[2] * self.dt
            
            # update position
            m.p[0] += m.v[0] * self.dt
            m.p[1] += m.v[1] * self.dt
            m.p[2] += m.v[2] * self.dt
        
        ### update spring lengths
        for s in self.Springs:
            s.refresh_L()
            s.set_L_1(t)
           
        if verbose:
            for m in self.Masses:
                print(f"m.F = {m.F}, m.a = {m.a}, m.v = {m.v}, m.p = {m.p}")
                
        # return eneergies [kinetic, spring potential (including bounds), grav potential]
        return [self.kinetic, self.potential_springs, self.potential_gravity]

    def get_points(self):
        points = []
        for m in self.Masses:
            points.append(m.listify())
        return points
    
    def display_frame(self, save=False, filename=None):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.cla()

        ax.axes.set_xlim3d(left=0, right=self.box_dims[0]) 
        ax.axes.set_ylim3d(bottom=0, top=self.box_dims[1]) 
        ax.axes.set_zlim3d(bottom=0, top=self.box_dims[2]) 
        
        # scatter plot of Point Masses
        for m in self.Masses:
            ax.scatter3D(m.p[0], m.p[1], m.p[2], color='k')
        
        # plot of Springs
        for s in self.Springs:
            x = [s.m1.p[0], s.m2.p[0]]
            y = [s.m1.p[1], s.m2.p[1]]
            z = [s.m1.p[2], s.m2.p[2]]
            if s.status == 'steady':
                color = 'green'
            elif s.status == 'stretched':
                color = 'red'
            else:
                color = 'blue'
            ax.plot3D(x, y, z, color=color)

        
        # display or save
        if save:
            fig.savefig(filename)
            
        plt.close()
    
    def simulate(self, t, save = False, filename = '', verbose=False):
        """
        t: list of time points, spaced by dt
        
        filename: template for beginning of filename
        
        returns a list of 
        """
        
        length = len(t)
        
        digit_length = len(str(length))
        
        for i, t_ in enumerate(t):
            # do integration step
            energies = self.integration_step(t=t_, verbose=verbose)
            
            self.energies.append(energies)
            
            # get filename
            i_str = str(i)
            num_digits = len(i_str)
            
            frame_filename = filename + '0'*(digit_length - num_digits) + i_str
            
            self.display_frame(save=save, filename=frame_filename)

            self.points.append([m.p.copy() for m in self.Masses])
            
        return self.points, self.energies
