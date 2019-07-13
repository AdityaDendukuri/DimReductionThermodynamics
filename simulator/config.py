#impory numerical calculation library
import numpy as np 

#Import and prepare plot 
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


#function to generate fcc cube 
def fcc(sigma):
      config = []
      origin = [0.0, 0.0, 0.0]
      config.append(origin)
      seperation = 2.5 * sigma + 2.0
      for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                  for k in range(-1, 2, 1):
                        #ignore edges
                        if abs(i) == abs(j) and k == 0:
                              continue
                        if abs(j) == abs(k) and i == 0:
                              continue
                        if abs(k) == abs(i) and j == 0:
                              continue
                        if i == 0 and k == 0 and abs(j) == 1:
                              continue
                        #add to configurations
                        config.append([origin[0] + i*seperation, origin[1] + j*seperation, origin[2] + k*seperation])
      return config

#generate initial positions
init_positions = fcc(0.003)

#plot the configiuration to visualize 
x_plt = []
y_plt = []
z_plt = []
for i in range(len(init_positions)):
      x_plt.append(init_positions[i][0])
      y_plt.append(init_positions[i][1])
      z_plt.append(init_positions[i][2])
ax.scatter(x_plt, y_plt, z_plt, s=20)
plt.show()


with open('config.txt', mode='w') as data:
      data.write(str(13) + ' \n')
      for i in range(len(init_positions)):
            data.write(str(i+1) + ' ' +str(init_positions[i][0]) + " " +  str(init_positions[i][1])+" " +str(init_positions[i][2]) + '\n')

