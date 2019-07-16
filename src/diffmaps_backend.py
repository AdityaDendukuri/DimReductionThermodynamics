'''
Dimensionality Reduction Using Diffusion Maps (backend)
Distance Metric based IsoRank based on: J. Phys. Chem. B 2014, 118, 15, 4228-4244

Author: 
      Aditya Dendukuri (adenduku@email.uark.edu)
      Department of Computer Science and Engineering
      University of Arkansas 

4/20/2019
'''

# import libraries
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


simulation_min = np.array([-5.0, -5.0, -5.0])
simulation_max = np.array([5.0, 5.0, 5.0])
side_length = (simulation_max - simulation_min)



class diff_map_solver:
      #initialize variables
      def __init__(self, constants):
            self.dim_H = constants['dim_H']
            self.constants = constants
            self.num_input = constants['num_data']
            self.sigma = 1.0
            self.num_particles = constants['num_particles']
            self.num_data = constants['num_data']
            #self.distance_range = [0.0, 1.3]
            self.distance_range = [0.0, 1.5]

      #euclidian distance with periodic boundary 
      def calc_distance(self, r1, r2):
            #manage periodic boundary 
            r12_ = [0.0,0.0,0.0]
            r12_[0] = (r1[0] - r2[0] + side_length[0]/2.) % side_length[0] - side_length[0]/2.
            r12_[1] = (r1[1] - r2[1] + side_length[1]/2.) % side_length[1] - side_length[1]/2.
            r12_[2] = (r1[2] - r2[2] + side_length[2]/2.) % side_length[2] - side_length[2]/2.
            return np.linalg.norm(r12_)
      
      #determine if two particles have a bond or not 
      def get_bond_value(self, max, r1, r2):
            dist = self.calc_distance(r1, r2)
            #return dist/max
            if dist > self.distance_range[0] and dist < self.distance_range[1]:
                  return 1
            else:
                  return 0 

      #Generate interparticle adjacency matrix 
      def generate_Rij(self, config):
            x = []
            particles = []
            num_particles = self.constants['num_particles']
            for i in range(0, len(config), num_particles):
                        particles.append(config[i:i+num_particles])
            for i in range(len(particles)):
                  a = []
                  for j in range(len(particles)):
                        a.append(self.calc_distance(particles[i], particles[j]))
                  x.append(a)
            return x

      #generate interparticle bond status = 1 or 0 (connected or not connected)
      def generate_Gij(self, config):
            x = []
            particles = []
            max = self.calc_max_dist(config)
            num_particles = self.constants['num_particles']
            for i in range(0, len(config), num_particles):
                        particles.append(config[i:i+num_particles])
            for i in range(len(particles)):
                  a = []
                  for j in range(len(particles)):
                        a.append(self.get_bond_value(max, particles[i], particles[j]))
                  x.append(a)
            return np.array(x)

      def calc_max_dist(self, config):
            x = []
            particles = []
            num_particles = self.constants['num_particles']
            for i in range(0, len(config), num_particles):
                        particles.append(config[i:i+num_particles])
            for i in range(len(particles)):
                  a = []
                  for j in range(len(particles)):
                        a.append(self.calc_distance(particles[i], particles[j]))
                  x.append(a)
            return np.max(x)

      #Calculate the adjacency matrices R (based on euclidian distance) and G (based on bond value: 0 or 1)
      @jit(nopython=True)
      def calculate_R_G(self, configurations):
            R = []
            G = []
            for i in range(len(configurations)):
                  #generate the adjacency matrices
                  R.append(self.generate_Rij(configurations[i]))
                  G.append(self.generate_Gij(configurations[i]))
            return np.array(R), np.array(G)

      #generate permutation matrices to rearrange R to optimize IsoRank 
      @jit(nopython=True)
      def generate_permutation_matrices(self, size):
            matrices = []
            I = np.identity(size)
            matrices.append(I)
            for m in itertools.permutations(I):
                  matrices.append(np.array(m))
            return matrices 
      
      #check similiarty between two
      def calc_similiarty(self, G_i, Gj):
            x = 0.0
            for a in range(len(G_i)):
                  for b in range(len(G_i[a])):
                        x += (G_i[a][b] - Gj[a][b])**2
            return np.sqrt(x)
      
      #shuffle data
      def shuffle(self, features, lables):
            x = np.arange(0, len(features))
            np.random.shuffle(x)
            new_features = []
            new_lables = []
            for i in range(len(x)):
                  new_features.append(features[x[i]])
                  new_lables.append(lables[x[i]])
            return new_features, new_lables

      #find index of an item 
      def index(self, array, item):
            for i in range(len(array)):
              if array[i] == item:
                  return i
            return 0

      #Modified IsoRank Algorithm 
      def isorank(self, configurations, R, G):
            R_ = np.array([])
            G_ = np.array([])
            scores = np.array([])
            #temporary array for intermediate steps 
            temp = np.array([])
            permut = self.generate_permutation_matrices(len(G[0]))  
            for i in range(len(configurations)):
                  scores.append([])
                  #Test similiarity with all the other configurations 
                  for j in range(len(configurations)):
                        #skip comparing with itself 
                        if j == i:
                              continue
                        temp.append([])
                        #Trying out all permutations
                        for k in range(len(permut)):
                              # G' = PGP'
                              G1_ = np.matmul(np.matmul(permut[k], G[i]), permut[k].transpose())
                              temp[i].append(G1_)
                              #similiarty criteria 
                              scores[i].append(self.calc_similiarty(G1_, G[j]))
            #Choosing the global minimum 
            for i in range(len(scores)):
                  idx = scores[i].index(np.min(scores[i]))
                  G_.append(temp[i][idx])
                  #rearrange R' with the most similiar arrangement
                  permut_idx = idx%len(permut)
                  R_.append(np.matmul(np.matmul(permut[permut_idx], R[i]), permut[permut_idx].transpose()))
            return R_, G_
      

      #function to compute d_ij for d matrix 
      @jit(nopython=True)
      def d_ij(self, R_i, G_i, Rj, Gj):
            x = 0.0
            num_particles = self.constants['num_particles']
            for a in range(num_particles):
                  for b in range(a, num_particles):
                        x += np.abs(R_i[a][b]*G_i[a][b]  - Rj[a][b]*Gj[a][b])
            return x
      
      def save_trajectories(self, x_plt, y_plt, z_plt, name):
            f1 = open(name + 'pc1', 'w')
            f2 = open(name + 'pc2', 'w')
            f3 = open(name + 'pc3', 'w')

            np.save(f1, x_plt)
            np.save(f2, y_plt)
            np.save(f3, z_plt)

            f1.close()
            f2.close()
            f3.close()

      def load_trajectories(self, name):
            f = open(name + 'pc1', 'w')
            a = np.load(name + 'pc1')
            b = np.load(name + 'pc2')
            c = np.load(name + 'pc3')
            f.close()
            return a, b, c


      def compute_d(self, R, G, R_, G_):
            d = np.zeros([self.num_input, self.num_input])
            for i in range(self.num_input):
                  for j in range(self.num_input):
                        d[i][j] = self.d_ij(R_[i], G_[i], R[j], G[j])
            return d

      def compute_A(self, d):
            A = np.array([])
            for i in range(len(d)):
                  A.append(np.array([]))
                  for j in range(len(d[i])):
                        A[i].append( np.exp(- (d[i][j]**2)/(2.0*self.sigma)))
            return A
      
      def compute_Dij(self, i, A):
           x = 0.0
           for j in range(len(A)):
                 x += A[i][j]
           return x
      
      #calculate diaognal matrix 
      @jit(nopython=True)
      def compute_D(self, A):
            D = np.array([])
            for i in range(len(A)):
                  D.append([])
                  for j in range(len(A[i])):
                        if i == j:
                              D[i].append(self.compute_Dij(i, A))
                        else:
                              D[i].append(0.0)
            return D

      def compute_M(self, D, A):
            return np.matmul(np.linalg.inv(D), A)

      def generate_eigenspace(self, M):
            return np.linalg.eig(M)

      def plot_explained_variance(self, eig_vals):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            tot = sum(eig_vals)
            var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
            cum_var_exp = np.cumsum(var_exp)
            xplot = np.arange(0, len(cum_var_exp))
            ax.bar(xplot, var_exp)
            ax.scatter(xplot, cum_var_exp)
            plt.legend(['Cumulative Explained Variance', 'Explained Variance'], loc='upper right')
            plt.savefig("exp_cuml_var.png")
            print("explained variance saved in exp_cuml_var.png!")

      def reload(self, name):
            x_plt, y_plt, z_plt = self.load_trajectories('0.18')
            return x_plt, y_plt, z_plt

      #load data
      def load_data(self, temperature):
            features = []
            labels = []
            file_to_parse = open('features.txt', "r")
            iter = 0
            lines = file_to_parse.readlines()
            for x in lines:
                if len(features) >= self.num_data:
                      break
                data = x.split()
                temp = []
                i=0
                if float(data[0]) != temperature:
                    continue
                for i in range(self.dim_H):
                    temp.append(float(data[i+1]))
                features.append(temp)
                labels.append(float(data[self.dim_H+1]))
                iter+=1 
            self.num_input = len(features)
            return np.reshape(features, [len(np.asarray(features)), self.dim_H]), np.reshape(labels, [len(np.asarray(labels)), 1])
      
      def plot_new_space_2d(self, eigvals, eigvecs, potentials):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            x_plt = []
            y_plt = []
            temp = []
            for i in range(len(eigvals)):
                  temp.append(eigvals[i])
            np.sort(temp)
            idx1 = self.index(eigvals, temp[1])
            idx2 = self.index(eigvals, temp[2])
            x_plt = []
            y_plt = []
            for i in range(len(eigvecs)):
                  x_plt.append(eigvecs[i][idx1])
                  y_plt.append(eigvecs[i][idx2])
            ax.scatter(x_plt, y_plt, c=potentials)
            plt.savefig("new_2d_space.png")
      
      def plot_2d(self, x_plt, y_plt, potentials):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.scatter(x_plt, y_plt, c=potentials)
            plt.savefig("new_2d_space.png")
      
      def plot_3d(self, x_plt, y_plt, z_plt, potentials):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_plt, y_plt, z_plt, c=potentials)
            plt.show()


      def plot_histogram(self, x, name):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.hist(np.reshape(x, len(x)), bins=100)
            plt.savefig(name)
            

      def map_potential_energy(self, eigvals, eigvecs , potentials):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            x_plt = []
            y_plt = []
            temp = []
            for i in range(len(eigvals)):
                  temp.append(eigvals[i])
            np.sort(temp)
            idx1 = self.index(eigvals, temp[1])
            idx2 = self.index(eigvals, temp[2])
            x_plt = []
            y_plt = []
            ax.set_xlabel('eig vec 1')
            ax.set_ylabel('potential energy')

            for i in range(len(eigvecs)):
                  x_plt.append(eigvecs[i][idx1])
                  y_plt.append(potentials[i])
            ax.scatter(x_plt, y_plt)
            plt.savefig("new_2d_space.png")
            plt.show()

      def plot_new_space_3d(self, eigvals, eigvecs, potentials):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')                        
            temp = []
            np.sort(temp)
            for i in range(len(eigvals)):
                  temp.append(eigvals[i])
            idx1 = self.index(eigvals, temp[1])
            idx2 = self.index(eigvals, temp[2])
            idx3 = self.index(eigvals, temp[3])
            x_plt = []
            y_plt = []
            z_plt = []
            for i in range(len(eigvecs)):
                  x_plt.append(np.array(eigvecs[i][idx1].real))
                  y_plt.append(np.array(eigvecs[i][idx2].real))
                  z_plt.append(np.array(eigvecs[i][idx3].real))
            self.save_trajectories(x_plt, y_plt, z_plt, '0.18')
            ax.scatter(np.array(x_plt), np.array(y_plt), np.array(z_plt), c=potentials[:self.num_data])
            plt.show()
      
     






























