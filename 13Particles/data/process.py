
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def plot_histogram(x):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.hist(np.reshape(x, len(x)), bins=100)
    plt.show()

num_particles = 13

coordinates = []
potentials = [] 

file_to_parse = open("T017.txt", "r")
file_to_write = open("13P_T017_RAW.txt", "w")

lines = file_to_parse.readlines()
iter = 0
for i, x in enumerate(lines):
      data = x.split()
      coordinates.append(float(data[0]))
      coordinates.append(float(data[1]))
      coordinates.append(float(data[2]))
      if i%(num_particles) == 0:
            potentials.append(float(data[3]))
   
coordinates = np.reshape(coordinates, [int(len(coordinates)/(num_particles*3)), num_particles*3])
print(len(potentials))


for i in range(len(potentials)):
      for j in range(len(coordinates[i])):
            file_to_write.write(str(coordinates[i][j]) + " ")
      file_to_write.write(str(potentials[i]) + " \n")

lowest_index = np.where(np.min(potentials))[0][0]
structure = np.reshape(coordinates[lowest_index], [num_particles, 3])


fig1 = plt.figure(figsize=(10, 8))
plt.title("T = 0.17, x:Predicted Potential Energy, y: Original Potential Energy")
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(structure[:,0], structure[:,1], structure[:,2], "-o")
plot_histogram(potentials)
