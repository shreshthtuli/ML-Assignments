"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Gaussian Discriminant Analysis

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
from random import shuffle, random

# Variables used in learing algorithm
iteration = 0
LearningRate = 0.1
Epsilon = 1e-30

# Data variables
X_orig = []
Y = []
X = [] # Normalized X with X0 as 1

# Read Data from file and return numpy array
def read_float(filename):
	return np.matrix([map(float, line.strip().split()) for line in open(filename)])

def read_string(filename):
    return np.matrix([map(str, line.strip().split()) for line in open(filename)])

# Returns normalised numpy array
def normalise(X):
	return (X - np.mean(X, axis=0))/np.std(X, axis=0)

def phi(X, Y):
    sum1 = 0
    sum0 = 0
    for i in Y:
        if i == 'Canada': sum1 += 1
        else: sum0 += 1
    return (sum1 / (sum1 + sum0))


# Read Data
X_orig = read_float('q4X.dat')
Y = read_string('q4Y.dat')

# Prints size of training set
print ("Number of examples : %s" % X_orig.shape[0])

# Normalize X values
X = normalise(X_orig)

# We consider Alaska as 0 and Canada as 1
# Calculate all parameters
u0 = np.zeros((1, 2))
u1 = np.zeros((1, 2))
sum1 = 0.0
sum0 = 0.0

for i in range(len(X)):
    if Y.item(i) == 'Canada': # Case 1
        sum1 += 1
        u1 += X[i]
    else: # Case 0
        sum0 += 1
        u0 += X[i]

phi = sum1 / (sum1 + sum0)
u0 = u0 / sum0
u1 = u1 / sum1

sigma0 = np.zeros((2, 2))
sigma1 = np.zeros((2, 2))

for i in range(len(X)):
	if Y.item(i) == 'Canada':
		sigma1 += np.outer((X[i] - u1), (X[i] - u1))
	else:
		sigma0 += np.outer((X[i] - u0), (X[i] - u0))

sigma = (sigma1 + sigma0) / X.shape[0]
sigma1 /= sum1
sigma0 /= sum0

print 'Phi = ', phi
print 'u0 = ', u0
print 'u1 = ', u1
print 'Sigma = \n', sigma
print 'Sigma0 = \n', sigma0
print 'Sigma1 = \n', sigma1

print X

for place in ('Alaska', 'Canada'):
    indices = np.where(np.c_[X, Y][:, 2] == place)
    Xtemp = X[indices[0]]
    Xp = Xtemp[:, 0]
    Yp = Xtemp[:, 1]
    plt.plot(Xp, Yp, 'ro', label=place, c='r' if place == 'Alaska' else 'b')


plt.show()