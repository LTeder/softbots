import torch
import numpy as np
import matplotlib.pyplot as plt

from bots import Spring, Mass, Universe


Masses = [Mass(m=0.1, p=[2, 2, 0], v=[0,0,0]),
          Mass(m=0.1, p=[5, 2, 0], v=[0,0,0]),
          Mass(m=0.1, p=[2, 5, 0], v=[0,0,0]),
          Mass(m=0.1, p=[5, 5, 0], v=[0,0,0]),
         
          Mass(m=0.1, p=[2, 2, 3], v=[0,0,0]),
          Mass(m=0.1, p=[5, 2, 3], v=[0,0,0]),
          Mass(m=0.1, p=[2, 5, 3], v=[0,0,0]),
          Mass(m=0.1, p=[5, 5, 3], v=[0,0,0])]

Springs = [Spring(Masses, L_0=3, k=2000, m1_idx=0, m2_idx=1, omega=1, c=0.25),
           Spring(Masses, L_0=3, k=1500, m1_idx=1, m2_idx=3, omega=1, c=0.25),
           Spring(Masses, L_0=3, k=1000, m1_idx=0, m2_idx=2, omega=1, c=0.25),
           Spring(Masses,L_0=3, k=1500, m1_idx=2, m2_idx=3, omega=1, c=0.25),
           
           Spring(Masses, L_0=3, k=2000, m1_idx=4, m2_idx=5, omega=1, c=0.25), 
           Spring(Masses, L_0=3, k=1500, m1_idx=5, m2_idx=7, omega=1, c=0.25),
           Spring(Masses, L_0=3, k=1000, m1_idx=4, m2_idx=6, omega=1, c=0.25),
           Spring(Masses,L_0=3, k=1500, m1_idx=6, m2_idx=7, omega=1, c=0.25),
           
           Spring(Masses, L_0=3, k=2000, m1_idx=0, m2_idx=4, omega=1, c=0.25), 
           Spring(Masses, L_0=3, k=1500, m1_idx=1, m2_idx=5, omega=1, c=0.25),
           Spring(Masses, L_0=3, k=1000, m1_idx=2, m2_idx=6, omega=1, c=0.25),
           Spring(Masses,L_0=3, k=1500, m1_idx=3, m2_idx=7, omega=1, c=0.25),
          
          # inside cross edges
           Spring(Masses, L_0=27**0.5, k = 5000, m1_idx= 0, m2_idx = 7, omega=1, c=0.25),
           Spring(Masses, L_0=27**0.5, k = 5000, m1_idx = 3, m2_idx = 4, omega=1, c=0.25),
           Spring(Masses, L_0=27**0.5, k = 5000, m1_idx= 1, m2_idx = 6, omega=1, c=0.25),
           Spring(Masses, L_0=27**0.5, k = 5000, m1_idx = 5, m2_idx = 2, omega=1, c=0.25)]

dt = 0.0005
uni = Universe(Masses, Springs, dt, grav=True)

t = np.arange(0, 3.0, dt)
points, energies = uni.simulate(t, save = False, filename='Crossed3DCube/frame', verbose=False)

points_tensor = torch.tensor(points)
torch.save(points_tensor, 'BreathingCube3D_dt0.0005.pt')

# plot energies
kinetic = [e[0] for e in energies]
potential_springs = [e[1] for e in energies]
potential_gravity = [e[2] for e in energies]
sum = []
for i in range(len(t)):
    sum.append(kinetic[i] + potential_springs[i] + potential_gravity[i])

plt.plot(t, kinetic, label = 'kinetic')
plt.plot(t, potential_springs, label = 'potential (springs)')
plt.plot(t, potential_gravity, label = 'potential (gravity)')
plt.plot(t, sum, label = 'total energy')
plt.legend()

plt.xlabel('Time (s)')
plt.title('Energy graph (breathing cube, $\omega = 1, c = 0.25$)')
plt.ylabel('Energy (Joules)')

fig = plt.gcf()
fig.set_size_inches(15, 10)