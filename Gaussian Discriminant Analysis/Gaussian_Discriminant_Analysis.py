"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Gaussian Discriminant Analysis

"""

import numpy as np
from numpy.linalg import inv
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

def boundary_expression(x, u0, u1, sigma0, sigma1, phi):
    u0 = u0.reshape(2, 1)
    u1 = u1.reshape(2, 1)
    sigma0 = np.matrix(sigma0)
    sigma1 = np.matrix(sigma1)
    expression = np.float64(((x - u1).T * sigma1.I * (x - u1)) / 2) - \
        np.float64(((x - u0).T * sigma0.I * (x - u0)) / 2) - \
        np.log(phi / (1 - phi)) + (np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0))) / 2
    
    return expression

def give_separator(x1, x2, u0, u1, sigma0, sigma1, phi):
    z = np.zeros(x1.shape)

    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
            z[i][j] = boundary_expression(x, u0, u1, sigma0, sigma1, phi)
    return z

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

# Plot data points

for place in ('Alaska', 'Canada'):
    indices = np.where(np.c_[X, Y][:, 2] == place)
    Xtemp = X[indices[0]]
    Xp = Xtemp[:, 0]
    Yp = Xtemp[:, 1]
    plt.subplot(1, 2, 1)
    plt.plot(Xp, Yp, 'ro', label=place, c='r' if place == 'Alaska' else 'b')
    plt.subplot(1, 2, 2)
    plt.plot(Xp, Yp, 'r.', label=place, c='r' if place == 'Alaska' else 'b')

plt.suptitle('Gaussian Discriminant Analysis')

# Plot separators : zoomed in
i = 3
plt.subplot(1, 2, 1)
x1 = np.arange(-i, i, 0.05)
x2 = np.arange(-i, i, 0.05)
x1, x2 = np.meshgrid(x1, x2)
# Linear Separator
z1 = give_separator(x1, x2, u0, u1, sigma, sigma, phi)
linear = plt.contour(x1, x2, z1, levels=[0], colors='green')
# Quadratic Separator
z2 = give_separator(x1, x2, u0, u1, sigma0, sigma1, phi)
quadratic = plt.contour(x1, x2, z2, levels=[0], colors='purple')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(loc='best')
plt.title('Zoomed in')

i = 10
plt.subplot(1, 2, 2)
x1 = np.arange(-i, i, 0.05)
x2 = np.arange(-i, i, 0.05)
x1, x2 = np.meshgrid(x1, x2)
# Linear Separator
z1 = give_separator(x1, x2, u0, u1, sigma, sigma, phi)
linear = plt.contour(x1, x2, z1, levels=[0], colors='green')
# Quadratic Separator
z2 = give_separator(x1, x2, u0, u1, sigma0, sigma1, phi)
quadratic = plt.contour(x1, x2, z2, levels=[0], colors='purple')
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(loc='best')
plt.title('Zoomed out')

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig('Gaussian-Discriminant-Analysis.png')
plt.show()