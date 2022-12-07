import matplotlib.pyplot as plt
import numpy as np
import timeit

from matplotlib import rc
rc('animation', html='jshtml')

from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

## functions
def complexity(heap):
    c=0
    for elem in heap:
        if elem != None:
            c += 1
    return c

def gaussian(x, mu = 0, sig = 1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def evaluate_single(heap_original, x, y, z, gauss_sigma = 3, mod_base = 2):
    """
    heap: binary heap representation of expression
    x: x value
    y: y value from 0
    z: z value from 0
    
    returns expression value
    """
    heap = heap_original.copy()
    
    for idx in range(len(heap)-1, 0, -1): # start from end
        if heap[idx] == None:
            pass
        elif heap[idx] == 'x':
            heap[idx] = x
        elif heap[idx] == 'y':
            heap[idx] = y
        elif heap[idx] == 'z':
            heap[idx] = z
        else:
            if 2*idx <= len(heap)-1:
                if heap[idx] == '+':
                    heap[idx] = heap[2*idx] + heap[2*idx+1]
                elif heap[idx] == '-':
                    heap[idx] = heap[2*idx] - heap[2*idx+1]
                elif heap[idx] == '*':
                    heap[idx] = heap[2*idx] * heap[2*idx+1]
#                 elif heap[idx] == '/':
#                     if heap[2*idx+1] == 0:
# #                         raise Exception('Divide by Zero')
#                         heap[2*idx+1] = 0.00001 # replace 0
#                     heap[idx] = heap[2*idx] / heap[2*idx+1]
                    
                elif heap[idx] == 'cos':
                    if heap[2*idx] == None:
                        heap[idx] = np.cos(heap[2*idx+1])
                    elif heap[2*idx+1] == None:
                        heap[idx] = np.cos(heap[2*idx])
                    else:
                        raise Exception("Unary operator has more than one operand")
                        
                elif heap[idx] == 'sin':
                    if heap[2*idx] == None:
                        heap[idx] = np.sin(heap[2*idx+1])
                    elif heap[2*idx+1] == None:
                        heap[idx] = np.sin(heap[2*idx])
                    else:
                        raise Exception("Unary operator has more than one operand")

                elif heap[idx] == 'gauss':
                    if heap[2*idx] == None:
                        heap[idx] = gaussian(heap[2*idx+1], 0, gauss_sigma)
                    elif heap[2*idx+1] == None:
                        heap[idx] = gaussian(heap[2*idx], 0, gauss_sigma)
                    else:
                        raise Exception("Unary operator has more than one operand")
                
                elif heap[idx] == 'mod':
                    if heap[2*idx] == None:
                        heap[idx] = heap[2*idx+1] % mod_base
                    elif heap[2*idx+1] == None:
                        heap[idx] = heap[2*idx] % mod_base
                    else:
                        raise Exception("Unary operator has more than one operand")

    return heap[1]             

def check_valid_heap(heap):
    """
    check: 
        unary operators only have one operand
        binary operators have 2 operands
        constants must be leaves
    """
    
    for idx in range(1, len(heap)):
        has_first_child = False
        if 2*idx < len(heap):
            if heap[2*idx] != None:
                has_first_child = True
                
        has_second_child = False
        if 2*idx+1 < len(heap):
            if heap[2*idx+1] != None:
                has_second_child = True
        
        
        # check binary
        if heap[idx] in ['+', '*', '-']:
            # missing both operators
            if not (has_first_child and has_second_child):
                return False
        
        # check unary
        if heap[idx] in ['sin', 'cos', 'gauss', 'mod']:
            # no operands
            if not (has_first_child or has_second_child):
                return False
            # two operands
            elif has_first_child and has_second_child:            
                return False
        
        # check constants
        if heap[idx] in ['x','y','z'] or type(heap[idx]) == float or type(heap[idx]) == int:
            # if there are any children
            if has_first_child or has_second_child:
                return False
    return True

def generate_heap_recursive(heap_depth, constant_max):
    heap_size = 2**heap_depth
    
    heap = [None] * (heap_size)
    
    def helper(elem, idx, heap, heap_depth, constant_max):
        """
        elem: node to be set
        idx: idx of node
        heap: list
        heap_depth: int
        constant_max: float
        """
        # set element
        if elem == 'constant':
            heap[idx] = np.random.uniform(-constant_max, constant_max)
        elif elem == 'xyz':
            heap[idx] = np.random.choice(['x', 'y', 'z'])
        else:
            heap[idx] = elem
        
        ## stop condition - leaf
        if idx >= 2**(heap_depth-1):
            pass            
        ## stop condition 2 - None, 'x', or 'constant'
        elif elem in [None, 'x', 'y', 'z', 'constant']:
            pass
            
        ## recursive call to children - limit valid children
        else: 
            ## set children element(s), depending on what parent elem is
            # binary operands
            if elem in ['+', '*', '-']:
                # if next nodes are leaves
                if 2*idx >= 2**(heap_depth-1):
                    # child 1
                    child1 = np.random.choice(['constant', 'xyz'])
                    helper(child1, 2*idx, heap, heap_depth, constant_max)
                    # child 2
                    child2 = np.random.choice(['constant', 'xyz'])
                    helper(child2, 2*idx+1, heap, heap_depth, constant_max)
                # next nodes could be parents   
                else:
                    # child 1
                    child1 = np.random.choice(['constant', 'xyz', '+', '*', '-', 'sin', 'cos', 'gauss', 'mod'])
                    helper(child1, 2*idx, heap, heap_depth, constant_max)
                    # child 2
                    child2 = np.random.choice(['constant', 'xyz', '+', '*', '-', 'sin', 'cos', 'gauss', 'mod'])
                    helper(child2, 2*idx+1, heap, heap_depth, constant_max)
            
            # unary operands - note: remove constants, that would be boring (?)
            if elem in ['sin', 'cos', 'gauss', 'mod']:
                # if next nodes are leaves
                if 2*idx+1 >= 2**(heap_depth-1):
                    child = np.random.choice(['x', 'y', 'z'])
                # next nodes could be parents
                else:
                    child = np.random.choice(['xyz', '+', '*', '-', 'sin', 'cos', 'gauss', 'mod'])
                # pick child index
                child_idx = np.random.choice([2*idx, 2*idx+1])
                
                helper(child, child_idx, heap, heap_depth, constant_max)
                    
    # set root node
    root_elem = np.random.choice(['+', '*', 'sin', 'cos', 'gauss', 'mod'])
    # limit to higher-level structural concepts
    # root_elem = np.random.choice(['gauss'])

    helper(root_elem, 1, heap, heap_depth, constant_max)
    
    return heap

# test
def eval_cube(heap, N = 3):
    arr = np.zeros((N,N,N))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                arr[i, j, k] = evaluate_single(heap, i, j, k, gauss_sigma = 3, mod_base = 2)

    print(arr)

def index_to_pos(idx, N = 3, side_length = 3, z_axis = False):
    if z_axis:
        return (idx - N/2) * side_length + side_length/2 + (side_length*N) / 2
    else:
        return (idx - N/2) * side_length + side_length/2
    
def get_index_pos_dict(N = 3, side_length = 3, z_axis = False):
    index_pos_dict = {}
    for i in range(N):
        
        index_pos_dict[i] = index_to_pos(i, z_axis=z_axis)
    return index_pos_dict

def exists(heap, index_pos_dict, index_pos_dict_z = None, N = 3, threshold = 0,
           gauss_sigma = 3, mod_base = 3):
    """
    cube_centers: array of floats

    returns an N x N x N tensor of True/False
    """

    if index_pos_dict_z == None:
        index_pos_dict_z = index_pos_dict

    arr = np.zeros((N,N,N), dtype = bool)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                x = index_pos_dict[i]
                y = index_pos_dict[j]
                z = index_pos_dict_z[k]

                val = evaluate_single(heap, x, y, z, gauss_sigma = gauss_sigma,
                                      mod_base = mod_base)
                
                arr[i, j, k] = (val > threshold)

    return arr

class Mass:
    def __init__(self, m=1, p=[0,0,0], v=[0, 0, 0], a=[0, 0, 0], F=[0, 0, 0], grounded=False, damping=0):
        self.m = m
        self.p = p
        self.v = v
        self.a = a
        self.F = F
        self.grounded = grounded
        self.damping = damping
        
    def __str__(self):
        pass

    def listify(self):
        return [self.p]
    
def dist(p1, p2):
    """
    p1: (x1, y1, z1)
    p2: (x2, y2, z2)
    """
    return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

def dist_2d(p1, p2):
    return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def normalize_2d(p):
    x, y = p
    
    if x==y==0:
        return p
    
    mag = (x**2 + y**2)**0.5
    return [x/mag, y/mag]

def normalize(p):
    x, y, z = p
    
    if x==y==z==0:
        return p
    
    mag = (x**2 + y**2 + z**2)**0.5
    return [x/mag, y/mag, z/mag]

class Spring:
    def __init__(self, m1, m2, L_0=1, k=1, status='steady', damping = 0.0, b = 0, omega = 0, c = 0):
        """
        
        m1: index in associated Mass list
        m2: index in associated Mass list
        
        omega: frequency of breathing
        c: coefficient of multiplication for breathing
        
        status: element in ['steady', 'compressed', 'stretched'] --> used for plotting later
        """
        self.L_0 = L_0
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.status = status
        self.damping = damping
        self.b = b
        self.omega = omega
        self.c = c
        
        self.L = dist(self.m1.p, self.m2.p)
        
        self.set_L_1()
        
    def set_L_1(self, t = 0):
        self.L_1 = self.L_0 * (1 + self.b * np.sin(2*np.pi * self.omega * t + self.c))
          
    def refresh_L(self):
        self.L = dist(self.m1.p, self.m2.p)
        
    def force(self):
        """
        if L > L_1, stretched; apply contraction force (positive)
        if L < L_1, compressed; apply expanding force (negative)
        
        returns force magnitude (direction to be determined outside)
        """
        
        return self.k * (self.L - self.L_1) * (1-self.damping)
    
    def energy(self):
        
        return 1/2 * self.k * (self.L - self.L_1)**2 
        
        # # positive when compressed, negative when stretched
        # if self.L < self.L_1:
        #     return -e
        # else:
        #     return e

    
from matplotlib.projections.polar import Axes
class Universe:
    def __init__(self, Masses, Springs, dt, box_dims = [20, 20, 20], K_G=1e6, g=-9.812, damping = 0,
                 mu=1.0):
        self.Masses = Masses
        self.Springs = Springs
        self.dt = dt
        self.box_dims = box_dims # bounds
        self.K_G=K_G # resistance for walls
        self.g = g # gravitational constant
        self.damping = damping
        self.mu = mu # coefficient of friction
        
        self.DIMENSIONS = 3
        
        for s in self.Springs:
            s.damping = self.damping
            
#         self.ax = plt.axes(projection='3d')


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

    def center_of_mass_horizontal(self):
        total_mass = 0
        x_center = 0
        y_center = 0

        for m in self.Masses:
            total_mass += m.m
            x_center += m.p[0] * m.m
            y_center += m.p[1] * m.m

        x_center /= total_mass
        y_center /= total_mass

        return (x_center, y_center)


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
            m1_direction = normalize(m1_direction)
            m1_force = [m1_direction[axis] * magnitude for axis in range(self.DIMENSIONS)]
            
            s.m1.F[0] += m1_force[0] # x
            s.m1.F[1] += m1_force[1] # y
            s.m1.F[2] += m1_force[2] # z
            
            # force on m2
#             m2_direction = [s.m2.p[0] - s.m1.p[0], s.m2.p[1] - s.m1.p[1]]
            m2_direction = [s.m1.p[0] - s.m2.p[0], s.m1.p[1] - s.m2.p[1], s.m1.p[2] - s.m2.p[2]]

            m2_direction = normalize(m2_direction)
            m2_force = [m2_direction[axis] * magnitude for axis in range(self.DIMENSIONS)]
            
            s.m2.F[0] += m2_force[0] # x
            s.m2.F[1] += m2_force[1] # y
            s.m2.F[2] += m2_force[2] # z
            
            ### add spring potential energy
            self.potential_springs += s.energy()

        
        ### update Mass Forces
        for m in self.Masses:
            
            ### gravity
            m.F[2] += self.g * m.m
            
            # calculate gravitational potential energy
            self.potential_gravity += - self.g * m.m * m.p[2] # based on position 
            
            # calculate kinetic energy
            self.kinetic += 1/2 * m.m * (m.v[0]**2 + m.v[1]**2 + m.v[2]**2)

            ### boundary collision forces
            # # x dimension right wall
            # if m.p[0] > self.box_dims[0]:
            #     m.F[0] += self.K_G * (self.box_dims[0] - m.p[0])
                
            #     self.potential_springs += 1/2 * self.K_G * (self.box_dims[0] - m.p[0])**2
                
            # ground
            if m.p[2] < 0:
                # normal force
                normal_force = self.K_G * (0 - m.p[2])
                m.F[2] += normal_force
                # calculate elastic energy
                self.potential_springs += 1/2 * self.K_G * (0 - m.p[2])**2

                ## calculate friction force
                if not (m.v[0] == 0 and m.v[1] == 0):
                    # calculate direction of movement
                    x_normed, y_normed = normalize_2d(m.v[:2]) 
                    # oppose x direction
                    m.F[0] -= x_normed * self.mu * normal_force

                    # oppose y direction
                    m.F[1] -= y_normed * self.mu * normal_force
            
            # # y dimension left wall
            # if m.p[1] < 0:
            #     m.F[1] += self.K_G * (0 - m.p[1])
                
            #     self.potential_springs += 1/2 * self.K_G * (0 - m.p[1])**2
                
            # # y dimension right wall
            # if m.p[1] > self.box_dims[1]:
            #     m.F[1] += self.K_G * (self.box_dims[1] - m.p[1])
                
            #     self.potential_springs += 1/2* self.K_G * (self.box_dims[1] - m.p[1])**2
        
            # # x dimension left wall
            # if m.p[0] < 0:
            #     m.F[0] += self.K_G * (0 - m.p[0])
                
            #     self.potential_springs += 1/2 * self.K_G * (0 - m.p[0])**2
        
            # # ceiling
            # if m.p[2] > self.box_dims[2]:
            #     m.F[2] += self.K_G * (self.box_dims[2] - m.p[2])
                
            #     self.potential_springs += 1/2 * (self.box_dims[2] - m.p[2])**2
        

            ### (to-do: add additional forces)
            
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
           
        #
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
        
        # plot bounds
#         corner1 = [0, 0]
#         corner2 = [self.box_dims[0], 0]
#         corner3 = [0, self.box_dims[1]]
#         corner4 = self.box_dims
        
#         plt.plot()
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

        # increase size
        fig.set_size_inches(10, 10)

        # display or save
        if save:
            fig.savefig(filename)
            
        # plt.close()

        return ax
    
    def simulate(self, t, save = False, filename = '', verbose=False, animate=False,
                 nth_frame = 100):
        """
        t: list of time points, spaced by dt
        
        filename: template for beginning of filename
        
        returns a list of 
        """
        length = len(t)
        
        digit_length = len(str(length))

        frames = []

        start_pos_horizontal = self.center_of_mass_horizontal()
        
        for i, t_ in enumerate(t):
            # do integration step
            energies = self.integration_step(t=t_, verbose=verbose)
            
            self.energies.append(energies)
            
            # get filename
            i_str = str(i)
            num_digits = len(i_str)
            
            frame_filename = filename + '0'*(digit_length - num_digits) + i_str
            
            ## frame
            if animate:
                if i % nth_frame == 0:
                    frames.append(self.display_frame(save=save, filename=frame_filename))

            # don't display frame each time
            # self.display_frame(save=save, filename=frame_filename)

            self.points.append([m.p.copy() for m in self.Masses])

        if animate:
            anim = self.animate(frames)
        else:
            anim = None

        end_pos_horizontal = self.center_of_mass_horizontal()

        total_dist_horizontal = dist_2d(start_pos_horizontal, end_pos_horizontal)
            
        return self.points, self.energies, anim, total_dist_horizontal

    def animate_single_frame(self, ax):
        return ax

    def animate(self, frames):

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')

        ax.axes.set_xlim3d(left=0, right=self.box_dims[0]) 
        ax.axes.set_ylim3d(bottom=0, top=self.box_dims[1]) 
        ax.axes.set_zlim3d(bottom=0, top=self.box_dims[2]) 

        frames = iter(frames)
        anim = animation.FuncAnimation(fig, self.animate_single_frame, frames=frames, blit=False, repeat=True)

        return anim
    
# function to find biggest "chunk"
def get_biggest_chunk(exists_mat):

    def helper(chunk, checked_matrix, exists_mat, indexes, shape):
        """

        """
        if not checked_matrix[tuple(indexes)]: # not already checked
            # mark indexes as checked
            checked_matrix[tuple(indexes)] = True

            if exists_mat[tuple(indexes)]:
                # add to chunk
                chunk.append(indexes)


                ## check next indexes, only going up (avoid infinite recursion?)
                # change x
                if indexes[0] < shape[0] - 1:
                    indexes_x_up = indexes.copy()
                    indexes_x_up[0] += 1
                    helper(chunk, checked_matrix, exists_mat, indexes_x_up, shape)

                if indexes[0] > 0:
                    indexes_x_down = indexes.copy()
                    indexes_x_down[0] -= 1
                    helper(chunk, checked_matrix, exists_mat, indexes_x_down, shape)

                # change y
                if indexes[1] < shape[1] - 1:
                    indexes_y_up = indexes.copy()
                    indexes_y_up[1] += 1
                    helper(chunk, checked_matrix, exists_mat, indexes_y_up, shape)

                if indexes[1] > 0:
                    indexes_y_down = indexes.copy()
                    indexes_y_down[1] -= 1
                    helper(chunk, checked_matrix, exists_mat, indexes_y_down, shape)
                
                # change z
                if indexes[2] < shape[2] - 1:
                    indexes_z_up = indexes.copy()
                    indexes_z_up[2] += 1
                    helper(chunk, checked_matrix, exists_mat, indexes_z_up, shape)

                if indexes[2] > 0:
                    indexes_z_down = indexes.copy()
                    indexes_z_down[2] -= 1
                    helper(chunk, checked_matrix, exists_mat, indexes_z_down, shape)

    shape = exists_mat.shape
    # define "already checked" matrix
    checked_matrix = np.zeros_like(exists_mat, dtype=bool)

    chunks = []

    # loop through true values in exists_mat and check adjacencies?
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):

                if exists_mat[i, j, k] == True:
                    # check if already checked
                    if not checked_matrix[i, j, k] == True:
                        # call helper function
                        chunk = []
                        helper(chunk, checked_matrix, exists_mat,
                                             [i, j, k], shape )
                        chunks.append(chunk)

    # or loop through adjacent units and see if true? Probably the former 

    



        # if is true and adjacnet, add to current chunk

        # if not adjacent, start finding a new chunk (?)

        # pass onto adjacent vertices

        # final: return list of indexes (?)

    # print(checked_matrix)
    # check adjacencies

    # sort chunks by longest length
    chunks.sort(key = lambda chunk: len(chunk), reverse = True) # descending

    if len(chunks) == 0:
        return None
    else:
        return chunks[0]
    
# function to get spring types
    # if voxel doesn't exist, leave as 0
def get_spring_types(heaps, chunk, index_pos_dict, 
                     index_pos_dict_z = None, N = 3,
           gauss_sigma = 3, mod_base = 2):
    """

    returns an N x N x N tensor of True/False
    """
    heap1, heap2, heap3 = heaps # spring types

    if index_pos_dict_z == None:
        index_pos_dict_z = index_pos_dict

    spring_types = np.zeros((N,N,N), dtype = int)

    for triple in chunk:
        i, j, k = triple

        x = index_pos_dict[i]
        y = index_pos_dict[j]
        z = index_pos_dict_z[k]

        val1 = evaluate_single(heap1, x, y, z, gauss_sigma = gauss_sigma,
                                mod_base = mod_base)
        val2 = evaluate_single(heap2, x, y, z, gauss_sigma = gauss_sigma,
                                mod_base = mod_base)
        val3 = evaluate_single(heap3, x, y, z, gauss_sigma = gauss_sigma,
                                mod_base = mod_base)
                
        # determine spring type
        max_val = max([val1, val2, val3])
        if max_val == val1:
            spring_type = 1
        elif max_val == val2:
            spring_type = 2
        elif max_val == val3:
            spring_type = 3
        spring_types[i, j, k] = spring_type

    return spring_types

class Soft_Spring(Spring):
    def __init__(self, m1, m2, L_0=1, k=1000, status='steady', damping = 0.0):
        
        super().__init__(m1, m2, L_0, k, status, damping, b=0, omega=0, c=0)

class ExpandContract(Spring):
    def __init__(self, m1, m2, L_0=1, k=5000, status='steady', damping = 0.0, b = 0.1, omega = 0.5):
        
        super().__init__(m1, m2, L_0, k, status, damping, b=b, omega = omega, c=0)
        
class ContractExpand(Spring):
    def __init__(self, m1, m2, L_0=1, k=5000, status='steady', damping = 0.0, b = 0.1, omega = 0.5):
        
        super().__init__(m1, m2, L_0, k, status, damping, b=b, omega = omega, c=np.pi)
        
spring_lib = {1: ContractExpand,
              2: ExpandContract,
              3: Soft_Spring} ### RUN THIS

# function to convert spring_types to Voxels (Springs + Masses)

# helper function to turn a single voxel into Springs and non-redundant Masses
    # idea: include Masses by default, but include a boolean flag for simulation
        # then only decide if springs connect or not

def get_voxels(spring_types, N = 3, default_mass = 1):

    shape = spring_types.shape

    # lower the cubes if the lowest z index is greater than 0
    lowest_z = N-1
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if spring_types[i, j, k] != 0:
                    if k < lowest_z:
                        lowest_z = k

    # note: can only do this once
    for z in range(lowest_z): # each time, lower the indexes by one
        for layer in range(N-1):
            spring_types[:,:,0+layer] = spring_types[:,:,1+layer]
        spring_types[:,:,-1] = np.zeros((N, N))

    ## define masses and outer springs
    masses = []
    springs = []
    # get mass matrix
    mass_matrix = [[[None for k in range(N+1)] for j in range(N+1)] for i in range(N+1)]
    spring_list = [] # list of ( (idx1), (idx2) ) and ( (idx2), (idx1) )

    # define masses and springs
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # check if exists
                spring_type = spring_types[i,j,k]
                if spring_type != 0:
                    ## define new masses
                    for x_add in range(2):
                        for y_add in range(2):
                            for z_add in range(2):
                                if mass_matrix[i+x_add][j+y_add][k+z_add] == None:
                                    p = [N*(i+x_add), N*(j+y_add), N*(k+y_add)]

                                    new_mass = Mass(default_mass, p = p)

                                    mass_matrix[i+x_add][j+y_add][k+z_add] = new_mass

                                    masses.append(new_mass)

                    ## define inner springs (x4)
                    inner_indexes = [
                    [(i, j, k) , (i+1, j+1, k+1)],
                    [(i, j, k+1), (i+1, j+1, k)],

                    [(i+1, j, k), (i, j+1, k+1)],
                    [(i, j+1, k), (i+1, j, k+1)]]

                    for pair in inner_indexes:
                        idx1, idx2 = pair
                        new_spring = spring_lib[spring_type](mass_matrix[idx1[0]][idx1[1]][idx1[2]],
                                                             mass_matrix[idx2[0]][idx2[1]][idx2[2]],
                                                             L_0 = N* (3**0.5) )
                        
                        springs.append(new_spring)
                        spring_list.append((idx1, idx2))
                        spring_list.append((idx2, idx1)) # get two orietnations

                    ## define outer springs (remaining connections)
                    outer_indexes = [
                        # bottom square
                        [(i, j, k), (i+1, j, k)], 
                        [(i, j, k), (i, j+1, k)],
                        [(i, j+1, k), (i+1, j+1, k)],
                        [(i+1, j, k), (i+1, j+1, k)],
                        # top square
                        [(i, j, k+1), (i+1, j, k+1)], 
                        [(i, j, k+1), (i, j+1, k+1)],
                        [(i, j+1, k+1), (i+1, j+1, k+1)],
                        [(i+1, j, k+1), (i+1, j+1, k+1)],
                        # vertical joining springs
                        [(i, j, k), (i, j, k+1)],
                        [(i, j+1, k), (i, j+1, k+1)],
                        [(i+1, j, k), (i+1, j, k+1)],
                        [(i+1, j+1, k), (i+1, j+1, k+1)]

                    ]

                    for pair in outer_indexes:
                        idx1, idx2 = pair # soft springs
                        # check if already in springs
                        if (idx1,idx2) in spring_list or (idx2,idx1) in spring_list:
                            pass
                        else:
                            new_spring = spring_lib[3](mass_matrix[idx1[0]][idx1[1]][idx1[2]],
                                                       mass_matrix[idx2[0]][idx2[1]][idx2[2]],
                                                        L_0 = N  )
                            
                            springs.append(new_spring)
                            spring_list.append((idx1, idx2))
                            spring_list.append((idx2, idx1)) 
                                

    return masses, springs

def generate_genome(heap_depth, constant_max):
    
    full_genome = []
    for i in range(4):
        heap = generate_heap_recursive(heap_depth, constant_max)
        full_genome.append(heap)
        
    return full_genome
    
def crossover(parent_1, parent_2, retry_bound = 50, verbose=True):
    """

    pick one node index in parent_1
        re-select until not None
        
    pick another node index in parent_2 of same depth
    
    swap, check if resulting heaps are valchild_1
    """
    def crossover_helper(heap_1, heap_2, idx_1, idx_2):
        # stop condition
        if idx_1 >= len(heap_1) or idx_2 >= len(heap_2):
            pass
        
        else:
            temp = parent_1[idx_1]
            heap_1[idx_1] = heap_2[idx_2]
            heap_2[idx_2] = temp

            crossover_helper(heap_1, heap_2, 2*idx_1, 2*idx_2)
            crossover_helper(heap_1, heap_2, 2*idx_1+1, 2*idx_2+1)
    
    while True:
        retry_count = 0
        while True:
            idx_1 = np.random.randint(1, len(parent_1))
            if parent_1[idx_1] != None:
                break
            else:
                retry_count +=1
            if retry_count >retry_bound: 
#                 if verbose:
#                     raise Exception('too many retries')
                return parent_1, parent_2 # just don't crossover lol

        idx_depth = int(np.log2(idx_1))
        
        # pick second parent with same depth
        retry_count = 0
        while True:
            idx_2 = np.random.randint(2**idx_depth, 2**(idx_depth+1))
            if parent_2[idx_2] != None:
                break
            else:
                retry_count +=1
            if retry_count >retry_bound: 
#                 if verbose:
#                     raise Exception('too many retries')
                return parent_1, parent_2

        # cross
        child_1 = [p for p in parent_1] # make copies
        child_2 = [p for p in parent_2] 
        
        crossover_helper(child_1, child_2, idx_1, idx_2)
        
        # check valid heap
        if check_valid_heap(child_1) and check_valid_heap(child_2):
            return child_1, child_2
        else:
            if verbose:
                print('not valid heap')
      
def big_crossover(bundled_parent1, bundled_parent2, retry_bound=50, verbose = False):
    
    """
    perform crossover on all 4 inner genes
    
    return bundled_child1, bundled_child2
    """
    parent1_exists, parent1_1, parent1_2, parent1_3 = bundled_parent1
    parent2_exists, parent2_1, parent2_2, parent2_3 = bundled_parent2
    
    child1_exists, child2_exists = crossover(parent1_exists, parent2_exists,verbose=verbose)
    child1_1, child2_1 = crossover(parent1_1, parent2_1, verbose=verbose)
    child1_2, child2_2 = crossover(parent1_2, parent2_2, verbose=verbose)
    child1_3, child2_3 = crossover(parent1_3, parent2_3, verbose=verbose)
    
    bundled_child1 = [child1_exists, child1_1, child1_2, child1_3]
    bundled_child2 = [child2_exists, child2_1, child2_2, child2_3]
    
    return bundled_child1, bundled_child2
    
def mutate(heap, mult_factor = 0.05, constant_max = 1, max_retries = 50, verbose=True):
    """
    heap: array representing binary heap for expression
    """
    mutated = False
    # choose index until elem is not None
    
    while True: # loop for valid heap
        
        retry_count = 0
        while not mutated:

            idx = np.random.randint(1, len(heap))
            while heap[idx] == None:
                idx = np.random.randint(1, len(heap))

            # if idx is in leaf position
            elem = heap[idx]

            ########################## additive mutations #################################
            # note: the order of these conditional statements matters
            if idx < int(len(heap)/8) and np.random.uniform()<0.5: # remaining depth >= 3
                
                if elem in ['x','y','z']:
                # replace x with (1+epsilon) * x - epsilon*sin(x) OR 
                    # replace x with (1+epsilon) * x - epsilon*cos(x)
                    func = np.random.choice(['sin', 'cos', 'gauss', 'mod'])
                    epsilon = np.random.uniform(-constant_max, constant_max)
                    
                    heap[idx] = '-'
                    heap[2*idx] = '*'
                    heap[2*idx+1] = '*'
                    heap[2*(2*idx)] = 1+epsilon
                    heap[2*(2*idx)+1] = elem
                    
                    heap[2*(2*idx+1)] = epsilon
                    heap[2*(2*idx+1)+1] = func
                    
                    heap[2*(2*(2*idx+1)+1)] = elem
                    
            elif idx < int(len(heap)/4) and np.random.uniform()<0.5: # remaining depth >= 2
                # if elem == x
                if np.random.uniform() > 0.5:
                    pass # do a simpler mutation
                
                if elem in ['x','y','z']:
                    # replace x with (1+epsilon) * x - epsilon
                    epsilon = np.random.uniform(-constant_max, constant_max)

                    heap[idx] = '-'
                    heap[2*idx] = '*'
                    heap[2*idx+1] = epsilon
                    heap[2*(2*idx) ] = 1+epsilon
                    heap[2*(2*idx) + 1] = elem

                    mutated=True
                    
#                     print('case1') # debugging

                elif type(elem) == float: # constant
                    # replace C with x*epsilon + (C-epsilon) 
                    epsilon = np.random.uniform(-mult_factor, mult_factor)

                    heap[idx] = '+'
                    heap[2*idx] = '*'
                    heap[2*idx+1] = epsilon
                    heap[2*(2*idx) ] = np.random.choice(['x', 'y', 'z'])
                    heap[2*(2*idx) + 1] = elem-epsilon

                    mutated=True
                    
#                     print('case2') # debugging

            elif type(elem) == float:
                # replace with C * (1-epsilon)
                epsilon = np.random.uniform(-mult_factor, mult_factor)

                heap[idx] = elem * (1+epsilon)

                mutated=True
                
#                 print('case3') # debugging

            elif idx < int(len(heap)/2): # remaining depth >= 1
                # if elem == x
                if elem == ['x','y','z']:
                    # replace with (1-epsilon) * x
                    epsilon = np.random.uniform(-mult_factor, mult_factor)
                    heap[idx] = '*'
                    heap[2*idx] = 1-epsilon
                    heap[2*idx+1] = elem

                    # replace with x - epsilon

                    mutated=True
                    
#                     print('case4') # debugging
            ##################################################################
    
    
            ################### (to-do) subtractive mutations ###################
            #####################################################################
            
            else:
                retry_count += 1
                if retry_count > max_retries:
                    if verbose:
                        print('max mutation retries reached')
                    break
        

        if check_valid_heap(heap):
            break
        else:
            if verbose:
                print('not valid heap')
        
    return heap

def big_mutate(bundled_genome):
    """
    mutate one inner genome
    """
    heap_idx = np.random.randint(4)
    
    mutated_heap = mutate(bundled_genome[heap_idx])
    
    bundled_genome[heap_idx] = mutated_heap
    
    return bundled_genome

### Genetic algorithm code
# based on evaluations
def similarity(bundled_heap1, bundled_heap2, N=3, gauss_sigma = 3, mod_base = 2):
    
    heap1_exists, heap1_1, heap1_2, heap1_3 = bundled_heap1
    heap2_exists, heap2_1, heap2_2, heap2_3 = bundled_heap2
    
    
    test_x = list(range(N))
    test_y = list(range(N))
    test_z = list(range(N))
    
    sim_denom = 0
    
    for x in test_x:
        for y in test_y:
            for z in test_z:
                # exists
                sim_denom += (evaluate_single(heap1_exists, x,y,z) - evaluate_single(heap2_exists, x,y,z))**2
                # spring type 1
                sim_denom += (evaluate_single(heap1_1, x,y,z) - evaluate_single(heap2_1, x,y,z))**2
                
                # sprring type 2
                sim_denom += (evaluate_single(heap1_2, x,y,z) - evaluate_single(heap2_2, x,y,z))**2
                
                # spring type 3
                sim_denom += (evaluate_single(heap1_3, x,y,z) - evaluate_single(heap2_3, x,y,z))**2
        
    if sim_denom == 0:
        print('identical')
        return np.inf
    else:
        return 1/sim_denom
    
def diversity(population):
    """
    computes (sum of 1/similarities)popsize
    """
    total_div = 0

    for i in range(len(population)):
        _, genome1 = population[i]

        for j in range(i+1, len(population)):
            _, genome2 = population[j]

            total_div += 1/similarity(genome1, genome2)

    total_div /= len(population)

    return total_div
    
def get_dist_helper(T, dt, genome, damping,
                    N = 3,
                   side_length = 3, 
                   threshold = 0.3):
    ### get exists_mat
    heap_exists, heap1, heap2, heap3 = genome

    index_pos_dict = get_index_pos_dict(N = 3, side_length = 3)
    index_pos_dict_z = get_index_pos_dict(N = 3, side_length = 3, z_axis = True)


    # exists_mat = exists(heap1, index_pos_dict, index_pos_dict_z)
    exists_mat = exists(heap_exists, index_pos_dict, threshold = threshold)
#     print('got pasts exists_mat')
    
    ### get biggest chunk
    biggest_chunk = get_biggest_chunk(exists_mat)
    if biggest_chunk == None:
        return 0
    elif len(biggest_chunk) == 0:
        return 0 # an empty phenotype ain't moving :)
    
#     print('got past biggest_chunk')
    
    ### heaps for spring types
    heaps = [heap1, heap2, heap3]
    
    spring_types = get_spring_types(heaps, biggest_chunk, index_pos_dict, 
                                index_pos_dict_z = index_pos_dict_z)
    
#     print('got past spring_types')
    
    ### get masses and springs
    masses, springs = get_voxels(spring_types)
    
#     print('got past get_voxels')
    
    ### define universe
    universe = Universe(masses, springs, dt, damping = damping)
    
#     print('got past universe')
    
    ### simulate

    t = np.arange(0, T, dt)
    points, energies, anim, total_dist = universe.simulate(t, save = False, filename='', verbose=False, animate = False,
                                    nth_frame = 200)
    

    return total_dist

### Crossover and anti-crowding
def genetic_programming(depth, N, pop_size, num_gens, T, dt = 0.0001, p = 0.5, mutat_prob = 0.05, damping=0.05, 
                        constant_max = 1):
    """
 
    """
    
    start = timeit.default_timer()

    # generate starting populations
    population = []

    pop_count = 0
    while pop_count < pop_size:
  
        new_genome = generate_genome(depth, constant_max)
        new_dist = get_dist_helper(T, dt, new_genome, damping)


        population.append( [new_dist, new_genome] )
        pop_count += 1

    # sorted descending
    population.sort(key = lambda x: x[0], reverse=True)

    size = pop_size # starting population size
    
    best_dist_list = []
    best_genome_list = []
    dist_list = []
    diversity_list = []
    
    consec_count = 0 # if exceeds T_consec, 
    for gen in range(num_gens):
        
        print('generation', gen)

        ### Mutation Probability HILL CLIMBING ###
        
        for mutation in range(int(mutat_prob*size)):
            mut_idx = np.random.randint(0, len(population))
        
            _, genome = population.pop(mut_idx)
            
            new_genome = big_mutate(genome)

            # print('mutation:')
            # print(new_genome)
            # print()

            new_dist = get_dist_helper(T, dt, genome, damping)
            population.append([new_dist, new_genome])


        population.sort(key=lambda x: x[0], reverse=True)
            
        ### RECOMBINATION ###
        
        # generate N offspring
        new_population = []

        for n in range(pop_size):
            
            # to-do: implement niching
            idx_1, idx_2 = np.random.choice(len(population), 2, replace=False)
            [parent_1_dist, parent_1], [parent_2_dist, parent_2] = population[idx_1], population[idx_2]


            offspring_1, offspring_2 = big_crossover(parent_1, parent_2)

            # print('crossover')
            # print(offspring_1, offspring_2)
            # print()

            offspring_1_dist = get_dist_helper(T, dt, offspring_1, damping)
            offspring_2_dist = get_dist_helper(T, dt, offspring_2, damping)

            
            parents = [(idx_1, parent_1, parent_1_dist),
                        (idx_2, parent_2, parent_2_dist)]

            # get better offspring
            if offspring_1_dist > offspring_2_dist:
                better_offspring = offspring_1
                better_offspring_dist = offspring_1_dist
            else:
                better_offspring = offspring_2
                better_offspring_dist = offspring_2_dist
                
            # deterministic un-crowding
            if better_offspring_dist > parent_1_dist and better_offspring_dist > parent_2_dist:
                # replace parent that's more similar
                parent_1_sim = similarity(better_offspring, parent_1)
                parent_2_sim = similarity(better_offspring, parent_2)
                
                if parent_1_sim > parent_2_sim: # replace parent 1
                    population[idx_1] = [better_offspring_dist, better_offspring]
                else:
                    population[idx_2] = [better_offspring_dist, better_offspring]
                    
                
            # else: skip replacement, print message
            else:
                print('skipped crossover')

        #     ## keep both offspring
        #     # print('starting crossover')
        #     offspring_1, offspring_2 = crossover(parent_1, parent_2)

        #     # print('crossover')
        #     # print(offspring_1, offspring_2)
        #     # print()

        #     offspring_1_dist = helper(offspring_1, spring_indexes, spring_lengths)
        #     offspring_2_dist = helper(offspring_2, spring_indexes, spring_lengths)

        #     new_population.append( [offspring_1_dist, offspring_1]) # stays sorted
        #     new_population.append( [offspring_2_dist, offspring_2])
        #     # print('crossover done')
        #     success=True
        # except Exception as e:
        #     print(e)
        #     print('Divide by zero error, retry')
            
        # if success: break
    
    # keep best offspring - ANTI CROWDING - replace parent if similar and better
#             dist_1 = tour_dist(points, offspring_1)
#             dist_2 = tour_dist(points, offspring_2)
#             if dist_1 < dist_2:
#                 new_population.add(offspring_1)
#             else:
#                 new_population.add(offspring_2)


        # population += new_population
        population.sort(key=lambda x: x[0],reverse=True)
        
        ### sort and remov bottom p fraction
        # WARNING: this takes the longest to execute
#         population.sort(key = lambda tour: tour_dist(points, tour)) # sort in ascending order

        #   population = population[:N] # keep best 50 %
        size = int(pop_size * p)
        ## pop remaining values
        for discard_ in range(len(population) - size):
            population.pop()
#         population = population[:size]
        
        
    
        ### update best dists, tours
        dist, genome = population[0]
        
        if gen == 0:
            best_dist = dist
            best_genome = genome
#             T_consec += 1
        
        elif dist > best_dist:
            best_genome = genome
            best_dist = dist
            
            # reset consecutive count
#             T_consec = 0
#         else:
#             T_consec += 1
#             if consec_count >= T_consec:
#                 print("ended early at t = {t}")
#                 break
        
        best_dist_list.append(best_dist)
        print(best_dist)

        best_genome_list.append(best_genome)
        print(best_genome)
        
        print('population size', len(population))
        print(population)
        
        diversity_list.append(diversity(population))
        
    end = timeit.default_timer()
    print(f"time elapsed: {end-start } s")
        
        # optional: update p
    return population, best_dist_list, best_genome_list, diversity_list