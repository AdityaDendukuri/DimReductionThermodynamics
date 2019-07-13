import numpy as np

#load data
def load_labels(path, dim_output):
    labels = []
    file_to_parse = open(path, "r")
    lines = file_to_parse.readlines()
    for x in lines:
        labels.append(float(x))
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

