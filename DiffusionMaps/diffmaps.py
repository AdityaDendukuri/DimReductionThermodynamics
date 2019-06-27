'''
DIFFUSION MAPS FOR REDUCING DIMENSIONS 
FOR MOLECULAR DYNAMICS (LENNARD JONES)

Author: Aditya Dendukuri
4/20/2019
'''
import diffmaps_backend 
import numpy as np

hyper_params = {
               #Higher dimension size
               'dim_H' : 9,
               #Density of the simulation (dimensionless)
               'density' : 0.003,
               #Num of particles in cluster
               'num_particles' : 3,
               #Number of configurations to be used 
               'num_data' : 100
               }
            
#initialize solver using the defined above  
diff_map_solver = diffmaps_backend.diff_map_solver(hyper_params)


'''
STEPS OF THE COMPLETE PROCESS
'''
#load data first according to temperature
phase_space, potentials = diff_map_solver.load_data(0.18)
#phase_space = phase_space_[0:diff_map_solver.num_data]
#phase_space = phase_space_[0:diff_map_solver.num_data]


#calculate R (distance tensor to hold matrices) and G 
R, G = diff_map_solver.calculate_R_G(phase_space)

print("R, G calculated")

#calculate R' and G' using modified isorank algorithm (selects best alignment)
R_, G_ = diff_map_solver.isorank(phase_space, np.array(R), np.array(G))

print("isorank done")

#calculate distance matrix = sum(|R'*G' - R*G|)
d = diff_map_solver.compute_d(R, G, R_, G_)

print("d calculated")

#compute A matrix (A_ij = e^(d_ij^2 / 2*sigma))
A = diff_map_solver.compute_A(d)

#compute D matrix
D = diff_map_solver.compute_D(A)

#compute M matrix = D' * A
M = diff_map_solver.compute_M(D, A)

print("Eigenstart")

#calculating eigen vectors
eig_vals, eig_vecs = diff_map_solver.generate_eigenspace(M)

print("Eigenend")

temp = []
for i in range(len(eig_vals)):
      temp.append(eig_vals[i])
np.sort(temp)

idx = diff_map_solver.index(eig_vals, temp[1])
f1 = open('classes.txt', 'w')
for i in range(len(eig_vecs)):
            f1.write(str(eig_vecs[i][idx])+"\n")
f1.close()

#plot everything
diff_map_solver.plot_explained_variance(eig_vals)
diff_map_solver.plot_new_space_2d(eig_vals, eig_vecs, np.reshape(potentials[:hyper_params['num_data']], hyper_params['num_data']))
diff_map_solver.plot_new_space_3d(eig_vals, eig_vecs, np.reshape(potentials[:hyper_params['num_data']], hyper_params['num_data']))
diff_map_solver.map_potential_energy(eig_vals, eig_vecs, np.reshape(potentials[:hyper_params['num_data']], hyper_params['num_data']))
#a, b, c = diff_map_solver.reload('0.18')
#diff_map_solver.plot_2d(a,b,potentials)
#diff_map_solver.plot_3d(a, b, c,potentials)

