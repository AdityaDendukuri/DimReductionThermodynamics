
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d(x, y, color = []):
      fig = plt.figure(figsize=(10, 8))
      ax = fig.add_subplot(111)
      if len(color) != len(x):
            ax.scatter(x, y)
      else:
            ax.scatter(x, y, c = color)
      plt.show()


def plot_3d(x, y, z, color = []):
      fig = plt.figure(figsize=(10, 8))
      ax = fig.add_subplot(111, projection='3d')
      if len(color) != len(x):
            ax.scatter(x, y, z)
      else:
            ax.scatter(x, y, z, c = color)
      plt.show()


def plot_histogram(x):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.hist(np.reshape(x, len(x)), bins=100)
    plt.show()     