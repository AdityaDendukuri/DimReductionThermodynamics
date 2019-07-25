from dataloader import *
from plotter import *
from neural_net import *
from scripts import *
from itertools import combinations 

#load dataset
coordinates = load_features("../data/13P_T029_CENTERED.txt", 13)
potentials = load_labels("../data/13P_T029_POTENTIALS.txt", 1)
num_data = len(coordinates)
num_particles = len(coordinates[0])/3 


structure = [39, 20, 20, 5, 2, 1]
activations = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
model_dir = "../models/13P_T029.ckpt"

nn = NEURAL_NET(13, structure, activations)
nn.learning_rate = 0.1
nn.train(coordinates, potentials, test_set_size=1000, num_steps=10000, save=True, restore=False, model_save_dir=model_dir)

reduced_space = nn.get_reduced_space()
output_space = nn.get_output_space()

f = open("space.txt", "w")

for i in range(len(reduced_space)):
        f.write(str(reduced_space[i][0]) + ", " + str(reduced_space[i][1]) + ", " + str(output_space[i]) + "\n")
