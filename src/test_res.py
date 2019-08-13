from residual_neural_net import *
from plotter import *
from dataloader import *
from scripts import *


coordinates = load_features("13P_CENTERED.txt", 13)
potentials = load_labels("13P_POTENTIALS.txt", 1)

structure = [39, 30, 20, 10, 5, 3, 1]
activations = ['tanh' for i in range(len(structure))]
res_structure = [[2, 3, 4, 5], [3, 4, 5], [4, 5], [5]]

nn = NEURAL_NET(13, structure, activations, res_structure)
