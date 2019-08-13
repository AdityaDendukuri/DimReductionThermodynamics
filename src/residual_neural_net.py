#Import Libraries 
import numpy as np
import tensorflow as tf
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import linalg
import csv

learning_rate = 0.01
temperature_to_load = 0.18
num_particles = 13
test_set_size = 500

class NEURAL_NET:
      def __init__(self, num_particles, structure, activations, res_structure):
            #setting constants
            self.num_particles = num_particles
            self.dim_H = structure[0]
            self.dim_L = structure[-2]
            self.dim_output = structure[-1]
            self.structure = structure
            self.activations = activations
            self.res_structure = res_structure
            self.learning_rate = 0.1
            self.current_loss = 0.01
            #DEFINING TENSORFLOW GRAPH BELOW
            self.graph = tf.Graph()
            with self.graph.as_default():
                  #define parameters (weights and biases)
                  self.weights = []
                  self.res_weights = []
                  self.biases = []
                  self.res_biases = []
                  #defining weights 
                  for i in range(len(structure)-1):
                        self.weights.append(tf.Variable(tf.random_normal(shape=[structure[i], structure[i+1]])))
                        self.biases.append(tf.Variable(tf.random_normal(shape=[1, structure[i+1]])))
                  for i in range(len(res_structure)):
                        self.res_weights.append(tf.Variable(tf.random_normal(shape=[structure[res_structure[i][0]], structure[res_structure[i][1]]])))
                        self.res_biases.append(tf.Variable(tf.random_normal(shape=[1, structure[res_structure[i][1]]] )))

                  #placeholders representing input and output 
                  self.H = tf.placeholder("float", [None, self.dim_H])
                  self.V = tf.placeholder("float", [None, self.dim_output])
                  # defining tensorflow computational graph nodes
                  self.reduce_dimension_node = self.reduce_dimension(self.H) 
                  self.calc_potential_node = self.predict_potential(self.reduce_dimension_node)  
                  # computational graph nodes to train model 
                  self.calc_loss_node = tf.losses.mean_squared_error(predictions=self.calc_potential_node, 
                                                                     labels=self.V)
                  #nodes for optimizers 
                  self.optimizer_node = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                               beta1=0.99, 
                                                               beta2=0.999, 
                                                               epsilon=1e-08)
                  #self.optimizer_node = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                  self.train_model_node = self.optimizer_node.minimize(self.calc_loss_node)
                  self.init_nn_node = tf.global_variables_initializer()
                  self.saver = tf.train.Saver()


      #train the neural network to learn based on lables 
      def train(self, features, labels, test_set_size = 100, restore = True, save = True, model_save_dir = '../models/T018_1.ckpt', num_steps = 10000, display_step = 1000):                               
            #split data for testing and training 
            train_features = features[:len(features)-test_set_size]
            train_lables = labels[:len(features)-test_set_size]
            test_features = features[len(features)-test_set_size:]
            test_lables = labels[len(features)-test_set_size:]
            #training using tf.session
            with tf.Session(graph = self.graph) as sess:
                  sess.run(self.init_nn_node)   
                  if restore:
                        self.saver.restore(sess, model_save_dir)
                  for step in range(1, num_steps+1):
                      sess.run(self.train_model_node, feed_dict={self.H:train_features, self.V:train_lables})
                      if step % display_step == 0 or step == 1:
                          self.loss = sess.run(self.calc_loss_node, feed_dict={self.H: test_features, self.V: test_lables})
                          print("Step " + str(step) + ", Loss = " + "{:.4f}".format(self.loss))
                  if save:
                        self.saver.save(sess, model_save_dir)
                  print("\n \n TRAINING COMPLETE. LOSS: {:.4f}".format(self.loss))

      def get_reduced_space(self, x, model_save_dir):
            with tf.Session(graph = self.graph) as sess:
                  self.saver.restore(sess, model_save_dir)
                  reduced_dimension = sess.run(self.reduce_dimension_node, feed_dict={self.H:x})
            return reduced_dimension

      def get_output_space(self, x, model_save_dir):
            with tf.Session(graph = self.graph) as sess:
                  self.saver.restore(sess, model_save_dir)
                  reduced_dimension = sess.run(self.calc_potential_node, feed_dict={self.H:x})
            return reduced_dimension
      '''
      def reduce_dimension(self, x):
            current = [x]
            for i in range(1, len(self.weights)):
                  current.append(np.zeros(shape=[1, self.structure[i]]))

            for i in range(1, len(self.weights)):
                current[i] += tf.add(tf.matmul(current[i-1], self.weights[i-1]), self.biases[i-1])
                for j in range(len(self.res_structure[i-2])):
                      current[self.res_structure[i-1][j]] += tf.matmul(current[i-1], self.res_weights[i-1][j])

            for i in range(1, len(current)-1):
                if self.activations[i] == 'tanh':
                    current[i] = tf.nn.tanh(current[i])
                    continue
                if self.activations[i] == 'sigmoid':
                    current[i] = tf.nn.sigmoid(current[i])
                    continue
                if self.activations[i] == 'relu':
                    current[i] = tf.nn.relu(current[i])
                    continue
                if self.activations[i] == 'softmax':
                    current[i] = tf.nn.softmax(current[i])
                    continue

            return current[len(current)-1]
      '''

      def reduce_dimension(self, x):
            current = [x]
            for i in range(1, len(self.structure)):
                  current.append(np.zeros(shape=[1, self.structure[i]]))
            for i in range(1, len(self.weights)):
                  current[i] += tf.add(tf.matmul(current[i-1], self.weights[i-1]), self.biases[i-1])
                  for j in range(len(self.res_structure)):
                        i_ = self.res_structure[j][0]
                        j_ = self.res_structure[j][1]
                        if i_ == i-1:
                              current[j_] += tf.add(tf.matmul(current[i_], self.res_weights[j]), self.res_biases[j])
                  if self.activations[i] == 'tanh':
                        current[i] = tf.nn.tanh(current[i])
                        continue
                  if self.activations[i] == 'sigmoid':
                        current[i] = tf.nn.sigmoid(current[i])
                        continue
                  if self.activations[i] == 'relu':
                        current[i] = tf.nn.relu(current[i])
                        continue
                  if self.activations[i] == 'softmax':
                        current[i] = tf.nn.softmax(current[i])
                        continue 
            return current[len(current)-2]



      def predict_potential(self, x):
            return tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])
      
      #generate weights using glorot 
      def glorot_init(self, shape):
          return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

      #randomly generate weights
      def classical_init(self, shape):
          return tf.random_normal(shape=shape)
            
            
            
            
            
            
            
            
            
