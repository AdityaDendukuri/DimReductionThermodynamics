from neural_net import *
from dataloader import *
from plotter import *


#load dataset
coordinates = load_features("../data/3P_CENTERED_COORDINATES.txt", 3)
potentials = load_labels("../data/3P_T018_POTENTIALS.txt", 1)
num_data = len(coordinates)
num_particles = len(coordinates[0])/3 

