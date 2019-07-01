import numpy as np

#load data
def load_lables(path, dim_output, temperature_to_load):
    labels = []
    file_to_parse = open(path, "r")
    lines = file_to_parse.readlines()
    for x in lines:
        data = x.split()
        if data[0] != str(temperature_to_load):
            continue
        labels.append(float(data[9+1]))
    return np.reshape(labels, [len(np.asarray(labels)), dim_output])

def load_features(path):
    features = []
    file_to_parse = open(path, "r")
    lines = file_to_parse.readlines()
    for x in lines:
        data = x.split(' ')
        temp = []
        i=0
        for i in range(9):
            temp.append(float(data[i]))
        features.append(temp)
    return np.reshape(features, [len(np.asarray(features)), 9])
