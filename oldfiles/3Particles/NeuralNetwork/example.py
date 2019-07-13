import numpy as np 
from dataloader import *
from boltzmann import NEURAL_NET

features = load_features('../data/features.txt')
lables = load_labels('../data/labels.txt', 1)


nn = NEURAL_NET(3, [9, 11, 5, 2, 1], ['tanh', 'tanh', 'tanh', 'tanh'])

nn.train(features, lables, test_set_size=100, restore=False)


