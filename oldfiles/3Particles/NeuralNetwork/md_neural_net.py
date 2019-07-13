'''
Project:
    Dimensionality Reduction for Thermodynamically
    Small Systems using Supervised Deep Learning.
Author: 
    Aditya Dendukuri
    Department of Computer Science and Engineering
    University of Arkansas

The neural network used is a Fully Connected Network 
'''

#Import Libraries 
import numpy as np
import tensorflow as tf
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg
import csv
from dataloader import *

#Import and prepare plot 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Define constants for training 
learning_rate = 0.1
num_steps = 10000
display_step = 1000
temperature_to_load = 0.18
num_particles = 3
test_set_size = 500

#define simulation space
simulation_min = np.array([-5.0, -5.0, -5.0])
simulation_max = np.array([5.0, 5.0, 5.0])
side_length = (simulation_max - simulation_min)

#model_save_dir = 'model_save/TransRot0.18.ckpt'
#model_save_dir = 'model_save/TransRot0.18_1.ckpt'
model_save_dir = 'model_save/TransRot0.18_2.ckpt'


#Network Charecteristics
dim_H = num_particles*3   #Higher Dimension Size       
dim_layer_1 = 11          #Hidden layer 1 size
dim_layer_2 = 4           #Hidden layer 2 size       
dim_L = 2                 #Reduced Dimension Size
dim_output = 1            #Output size (In this example output is potential energy (V))

def plot_histogram(x, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.hist(np.reshape(x, len(x)), bins=100)
    plt.savefig(name)

#generate weights using glorot 
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

#randomly generate weights
def classical_init(shape):
    return tf.random_normal(shape=shape)

#features are raw coordinates and 
#load data
labels = load_lables("../data/labels.txt", dim_output, temperature_to_load)
print(labels)
features = load_features("../data/features.txt")#new features are coordates with translational and rotational degrees of freedom removed

# Tensorflow Placeholders for the Higher Dimension (H), Potential Energies (V) 
H = tf.placeholder("float", [None, dim_H])
V = tf.placeholder("float", [None, dim_output])

# Split data for training and testing
train_features = features[:len(features)-test_set_size]
train_lables = labels[:len(features)-test_set_size]
test_features = features[len(features)-test_set_size:]
test_lables = labels[len(features)-test_set_size:]

#Definining Random Model (only for training) 
weights = {
    'w1': tf.Variable(classical_init([dim_H, dim_layer_1])),
    'w2': tf.Variable(classical_init([dim_layer_1, dim_layer_2])),
    'w3': tf.Variable(classical_init([dim_layer_2, dim_L])),
    'w4': tf.Variable(classical_init([dim_L, dim_output]))
}

biases = {
    'b1': tf.Variable(classical_init([1, dim_layer_1])),
    'b2': tf.Variable(classical_init([1, dim_layer_2])),
    'b3': tf.Variable(classical_init([1, dim_L])),
    'b4': tf.Variable(classical_init([1, dim_output]))
}
    
# Dimensionality Reduction Layer of Neural Net 
def reduce_dimension(x):
    x = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    x = tf.nn.tanh(x)
    x = tf.add(tf.matmul(x, weights['w2']), biases['b2'])
    x = tf.nn.tanh(x)
    x = tf.add(tf.matmul(x, weights['w3']), biases['b3'])
    x = tf.nn.tanh(x)
    return x

#Calculate Potential : output layer of neural net
def calculate_potential(x):
    x = tf.add(tf.matmul(x, weights['w4']), biases['b4'])
    return x

# defining tensorflow computational graph nodes
reduce_dimension_node = reduce_dimension(H) 
calc_potential_node = calculate_potential(reduce_dimension_node)  

# computational graph nodes to train model 
regularizer_node = tf.reduce_mean((tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3'])))
calc_loss_node = tf.losses.mean_squared_error(predictions=calc_potential_node, labels=V)
calc_loss_with_reg = calc_loss_node + 3.0 * regularizer_node

# nodes for optimizers 
optimizer_node = tf.train.AdamOptimizer(learning_rate=0.01, 
                                        beta1=0.4, 
                                        beta2=0.4, 
                                        epsilon=1e-08)
train_model_node = optimizer_node.minimize(calc_loss_node)

# node to intialize neural network
init_nn_node = tf.global_variables_initializer()

# tensorflow model saver 
saver = tf.train.Saver()

# Build graph by connecting nodes defined above
with tf.Session() as sess:
    sess.run(init_nn_node)   
    saver.restore(sess, model_save_dir)
    for step in range(1, num_steps+1):
        sess.run(train_model_node, feed_dict={H:train_features, V:train_lables})
        if step % display_step == 0 or step == 1:
            loss = sess.run(calc_loss_node, feed_dict={H: test_features, V: test_lables})
            print("Step " + str(step) + ", Loss = " + \
                  "{:.4f}".format(loss))
    # Control plotting data here 
    potentials = sess.run(calc_potential_node, feed_dict={H:features})
    print("standard deviation: ", np.std(potentials))  #outputs the standard deviation
    reduced_dimension = sess.run(reduce_dimension_node, feed_dict={H:features})
    saver.save(sess, model_save_dir)


#plotting 
fig = plt.figure(figsize=(5, 5))
fig1 = plt.figure(figsize=(5, 5))
fig2 = plt.figure(figsize=(5, 5))

ax2 = fig2.add_subplot(111)
ax1 = fig1.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

red_dim_x = reduced_dimension[:, 0]
red_dim_y = reduced_dimension[:, 1]

ax.set_xlabel('Reduced Variable 1')
ax.set_ylabel('Reduced Variable 2')

a = ax.scatter(red_dim_x, red_dim_y, s=5, c=labels.flatten())
c = ax2.scatter(red_dim_x, labels)
b = ax1.scatter()

fig.colorbar(a, label=' Potential Energy')

plt.tight_layout()
plt.show()
