import numpy as np
import itertools

def distance(x0, x1, dimensions=10.0):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def get_adjacent_matrices(config, max_bond):
    config = np.reshape(config, [int(len(config)/3), 3])
    R = np.zeros(shape=[len(config), len(config)])
    G = np.zeros(shape=[len(config), len(config)])
    for i in range(len(config)):
        for j in range(len(config)):
            R[i][j] = distance(config[i], config[j])
            if R[i][j] < max_bond and R[i][j] > 0:
                G[i][j]=1
    return R, G

    
def add_clusters(pointer, adj, current_node, clusters, clustered):
    #add clusters using the bond information calculated above 
    bond_space = adj[current_node]
    #print(clustered)
    if(np.sum(clustered) == len(adj)):
        return 
    for i in range(len(bond_space)):
        if i != current_node and clustered[i] == 0 and bond_space[i] == 1:
            clusters[pointer].append(i)
            clustered[i] = 1   
            add_clusters(pointer, adj, i, clusters, clustered)  

#build cluster map
def cluster_map(adj):
    clusters = []
    clustered = np.zeros(len(adj))
    pointer = 0
    for j in range(len(clustered)):
        if clustered[j] == 0:
            clusters.append([j])
            clustered[j] = 1
            add_clusters(pointer, adj, j, clusters, clustered)
            pointer += 1
    return clusters

#centroid using periodic average
def calc_centroid(config, size=10):
    #turn data into complex space
    norm = 2.*np.pi/size
    config = np.reshape(config, [int(len(config)/3), 3])
    config = np.multiply(config, 2.*np.pi/size)
    #calculate centroid using periodic average
    centroid = np.add(np.arctan2(-np.mean(np.sin(config), axis=0), -np.mean(np.cos(config), axis=0)), np.pi)/norm   
    return PBC(centroid)


def PBC(coordinates, size=10):
    return np.subtract(coordinates, np.round(np.divide(coordinates,size))*size)

#remove translational degrees of freedom
def center_dataset(coordinates, size=10):
    for i in range(len(coordinates)):
        config = np.reshape(coordinates[i], [int(len(coordinates[i])/3),3])
        config = np.add(config, size/2.)
        com = calc_centroid(coordinates[i])
        config[:,0] = np.subtract(config[:,0], com[0] + size/2.)
        config[:,1] = np.subtract(config[:,1], com[1] + size/2.)
        config[:,2] = np.subtract(config[:,2], com[2] + size/2.)
        coordinates[i] = np.reshape(config, [len(config)*3])
    return PBC(coordinates, size)


def align_dataset(coordinates, size=10):
    aligned_coordinates = np.zeros(shape=coordinates.shape)
    #define X, Y and Z axes 
    x1 = np.array([1.,0.,0.])
    x2 = np.array([0.,1.,0.])
    x3 = np.array([0.,0.,1.])
    i=0
    for config in coordinates:
        config = np.reshape(config, [int(len(config)/3), 3])
        X = config[:, 0]
        Y = config[:, 1]
        Z = config[:, 2]
        #calculating Moment of Intertia tensor (I)
        Ixx = np.sum(np.square(Y) + np.square(Z))
        Iyy = np.sum(np.square(X) + np.square(Z))
        Izz = np.sum(np.square(Y) + np.square(Y))
        Ixy = -np.sum(X*Y)
        Iyz = -np.sum(Y*Z)
        Izx = -np.sum(Z*X)
        I = np.array([[Ixx, Ixy, Izz],
                      [Ixy, Iyy, Iyz],
                      [Izx, Iyz, Izz]])
        #Calculate eigenspace of MOI tensor
        eigvals, eigvecs = np.linalg.eig(I)
        Q = np.array([[angle(eigvecs[0], x1), angle(eigvecs[0], x2), angle(eigvecs[0], x3)],
                      [angle(eigvecs[1], x1), angle(eigvecs[1], x2), angle(eigvecs[1], x3)],
                      [angle(eigvecs[2], x1), angle(eigvecs[2], x2), angle(eigvecs[2], x3)]])
        aligned_coordinates[i] = np.reshape([np.dot(Q, particle) for particle in config], [len(config)*3])
        i += 1
    return aligned_coordinates


def add_clusters(pointer, adj, current_node, clusters, clustered):
    #add clusters using the bond information calculated above 
    bond_space = adj[current_node]
    #print(clustered)
    if(np.sum(clustered) == len(adj)):
        return 
    for i in range(len(bond_space)):
        if i != current_node and clustered[i] == 0 and bond_space[i] == 1:
            clusters[pointer].append(i)
            clustered[i] = 1   
            add_clusters(pointer, adj, i, clusters, clustered)  

#build cluster map
def cluster_map(adj):
    clusters = []
    clustered = np.zeros(len(adj))
    pointer = 0
    for j in range(len(clustered)):
        if clustered[j] == 0:
            clusters.append([j])
            clustered[j] = 1
            add_clusters(pointer, adj, j, clusters, clustered)
            pointer += 1
    return clusters

mag = lambda r : np.linalg.norm(r)
cos_sim = lambda x, y : (x@y)/(mag(x)*mag(y))
cos_dist = lambda x, y : 1. - np.abs(cos_sim(x, y))

#centroid using periodic average
def calc_centroid(config, size=10):
    #turn data into complex space
    norm = 2.*np.pi/size
    config = np.reshape(config, [int(len(config)/3), 3])
    config = np.multiply(config, 2.*np.pi/size)
    #calculate centroid using periodic average
    centroid = np.add(np.arctan2(-np.mean(np.sin(config), axis=0), -np.mean(np.cos(config), axis=0)), np.pi)/norm   
    return PBC(centroid)


def PBC(coordinates, size=10):
    return np.subtract(coordinates, np.round(np.divide(coordinates,size))*size)


def angle(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def d_ij(R_i, G_i, Rj, Gj):
    x = 0.0
    num_particles = len(R_i)
    for a in range(num_particles):
          for b in range(a, num_particles):
                x += np.abs(R_i[a][b]*G_i[a][b]  - Rj[a][b]*Gj[a][b])
    return x

def isorank_dist(R, G, R_, G_):
    num_input=len(R)
    d = np.zeros([num_input, num_input])
    for i in range(num_input):
          for j in range(num_input):
                d[i][j] = d_ij(R_[i], G_[i], R[j], G[j])
    return d

def calc_rad_gyration(coordinates):
    rg = np.zeros(shape=len(coordinates))
    for i in range(len(rg)):
        config = np.reshape(coordinates[i], [int(len(coordinates[i])/3), 3])
        rg[i] = np.sum(np.square(config[i]))/len(config)
    return rg


def generate_permutation_matrices(size):
    matrices = []
    I = np.identity(size)
    matrices.append(I)
    for m in itertools.permutations(I):
          matrices.append(np.array(m))
    return matrices 

def isorank(R, G):
    R_ = []
    G_ = []
    scores = []
    #temporary array for intermediate steps 
    temp = []
    permut = generate_permutation_matrices(len(R[0]))
    for i in range(len(R)):
          scores.append([])
          #Test similiarity with all the other configurations 
          for j in range(len(R)):
                #skip comparing with itself 
                if j == i:
                      continue
                temp.append([])
                #Trying out all permutations
                for k in range(len(permut)):
                      # G' = PGP'
                      G1_ = permut[k] @ G[i] @ permut[k].transpose()
                      temp[i].append(G1_)
                      #similiarty criteria 
                      scores[i].append(np.linalg.norm(G1_- G[j]))
    #Choosing the global minimum 
    for i in range(len(scores)):
          idx = scores[i].index(np.min(scores[i]))
          G_.append(temp[i][idx])
          #rearrange R' with the most similiar arrangement
          permut_idx = idx%len(permut)
          R_.append(permut[permut_idx] @ R[i] @ permut[permut_idx].transpose())
    return R_, G_