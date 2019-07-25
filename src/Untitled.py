#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataloader import *
from plotter import *
from neural_net import *
from scripts import *
from itertools import combinations 


# In[2]:


#load dataset
coordinates = load_features("../data/13P_T029_CENTERED.txt", 13)
potentials = load_labels("../data/13P_T029_POTENTIALS.txt", 1)
num_data = len(coordinates)
num_particles = len(coordinates[0])/3 

x = []
y = []
z = []
c = []
for i in range(len(coordinates)):
    for j in range(0, len(coordinates[i]), 3):
        x.append(coordinates[i][j+0])
        y.append(coordinates[i][j+1])  
        z.append(coordinates[i][j+2])
        c.append(potentials[i])
plt.style.use('Solarize_Light2')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
a = ax.scatter(x, y, z, s=5, c=np.array(c).flatten())
fig.colorbar(a, label=' Potential Energy')
plt.show()


# In[3]:


plot_histogram(potentials)


# In[4]:


#calculating radius of gyration
RoG = np.zeros(len(coordinates))
for i in range(len(coordinates)):
    particles = np.reshape(coordinates[i], [num_particles, 3])
    for j in range(len(particles)):
        RoG[i] += np.linalg.norm(particles[j])/3.0


# In[29]:


#extract some information from the feature space 
side_length_space = [] #side lengths between each particles
perimeter = [] #side lengths between each particles
n_bonds = np.zeros(len(coordinates)) #number of bonds in the structure
cut_off_distance = 1.4 #sigma  
for i in range(len(coordinates)):
    side_length_space.append([])
    particles = np.reshape(coordinates[i], [num_particles, 3])
    perms = combinations(particles, 2) 
    for perm in perms:
        side_length_space[i].append(np.linalg.norm(perm[0] - perm[1]))   
    perimeter.append(np.sum(side_length_space[i]))
    for j in range(len(side_length_space[i])):
        if side_length_space[i][j] < cut_off_distance:
            n_bonds[i] += 1


# In[ ]:


structure = [39, 12, 5, 2, 1]
activations = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
model_dir = "../models/13P_T029.ckpt"

nn = NEURAL_NET(13, structure, activations)
nn.learning_rate = 0.1
nn.train(coordinates, potentials, test_set_size=1000, num_steps=10000, save=True, restore=False, model_save_dir=model_dir)


# In[ ]:





# In[ ]:




