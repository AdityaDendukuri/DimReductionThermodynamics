import numpy as np

#load data
def load_labels(path, dim_output):
    labels = []
    file_to_parse = open(path, "r")
    lines = file_to_parse.readlines()
    for x in lines:
        labels.append(float(x))
    return np.reshape(labels, [len(np.asarray(labels)), dim_output])

def load_features_2d(path, num_particles):
    features = []
    file_to_parse = open(path, "r")
    lines = file_to_parse.readlines()
    for x in lines:
        data = x.split()
        
        if len(data) != num_particles*3:
            continue
        temp = []
        i=0
        for i in range(num_particles*3):
            temp.append(float(data[i]))
        features.append(temp)
    return np.reshape(features, [len(np.asarray(features)), num_particles*3])

def load_dataset(path, n):
    coordinates = []
    potentials = []
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split()
            if i%n == 0:
                coordinates.append([])
                potentials.append(float(data[3]))
            coordinates[len(coordinates)-1].append(float(data[0]))
            coordinates[len(coordinates)-1].append(float(data[1]))
            coordinates[len(coordinates)-1].append(float(data[2]))
    return coordinates, potentials