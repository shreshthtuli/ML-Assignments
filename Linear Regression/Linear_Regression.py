"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Linear Regression

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import time
from random import shuffle, random

# Global variables
iteration = 0
LearningRate = 0.1
Epsilon = 1e-30
Saved_Theta = []
Saved_J = []

''' This provides the analytical solution '''
def analytical_solution(X, Y):
    return np.linalg.inv(X.T * X) * X.T * Y

''' To Read the X values '''
def Xread():
	return np.matrix([map(float, line.strip().split()) for line in open('linearX.csv')])

''' To Read the Y values '''
def Yread():
	return np.matrix([[float(line.strip())] for line in open('linearY.csv')])
 
''' To create a theta vector based on dimension of input '''
def initialize_theta(x):
	return np.matrix([[float(random())]] * (x))

''' Returns the value of J(theta) '''
def create_J(X, Y, Theta):
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

''' Returns the gradient of J(theta) '''
def create_gradJ(X, Y, Theta):
	return X.T * (X * Theta - Y) / X.shape[0]

''' Returns the gradient of J(theta) '''
def create_gradJ_sgd(X, Y, Theta, i):
	return X[i].T * ((X[i] * Theta) - Y[i]) 

''' Returns normalised data '''
def normalise(X):
	var = np.linalg.norm(X)
	mean = np.linalg.mean(X)
	return (X - mean) / var

''' The Gradient Descent Algorithm '''
def gradient_descent(X, Y):
	global iteration
	global LearningRate
	global Saved_Theta
	t = 0
	Theta = initialize_theta(X.shape[1])
	J = create_J(X, Y, Theta)
	Saved_J.append(J)
	Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
	while(True):
		# print LearningRate
		iteration += 1;
		start_time = time.time()
		gradJ = create_gradJ(X, Y, Theta)
		newTheta = Theta - LearningRate * gradJ
		newJ = create_J(X, Y, newTheta)
		Theta = newTheta
		t += (time.time() - start_time)
		if J - newJ < Epsilon: 	# Value has converged
			break		
		J = newJ
		Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
		Saved_J.append(J)
	print("Total time : "),
	print(t)
	return Theta

''' The Stochastic Gradient Descent Algorithm '''
def stochastic_gradient_descent(X, Y):
	global iteration
	global LearningRate
	global Saved_Theta
	t = 0
	Theta = initialize_theta(X.shape[1])
	J = create_J(X, Y, Theta)
	Saved_J.append(J)
	Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
	while(True):
		newTheta = Theta
		oldJ = J

		shuffled_list = range(1, X.shape[0])
		shuffle(shuffled_list)
		for x in shuffled_list:
			start_time = time.time()
			gradJ = create_gradJ_sgd(X, Y, Theta, x)
			newTheta = Theta - LearningRate * gradJ
			newJ = create_J(X, Y, newTheta)
			Theta = newTheta
			t += (time.time() - start_time)
			Saved_Theta.append([item for sublist in Theta.tolist() for item in sublist] + [J])
			Saved_J.append(J)
			J = newJ
			iteration += 1;

		if oldJ - newJ < Epsilon: 	# Value has converged
			break
	print("Total time : "),
	print(t)			
	return Theta

''' Equation of the hypothesis function '''
def linear(Theta, x):
	return (Theta.item(1) * x) + Theta.item(0)

def linear2(Theta, x):
	return (Theta[1] * x) + Theta[0]

### 3D plot of the J(theta) function ###
# Returns the value of J(theta)
def create_J_plot(Theta_0, Theta_1):
	Theta = np.matrix([[Theta_0],[Theta_1]])
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

# Read input values
X = Xread()
Y = Yread()

print ("Number of examples : %s" % X.shape[0])

# Take learning rate from user
print("Enter Learning rate : "),
LearningRate = input()

# Enter Gradient Descent Algorithm
print("Enter 0 for BGD and 1 for SGD : "),
option = input()

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean)/X_std
X = np.c_[np.ones((X.shape[0], 1)), X]

# Perform Gradient Descent
FinalTheta = []
if option == 1:
	FinalTheta = stochastic_gradient_descent(X, Y)
else:
	FinalTheta = gradient_descent(X, Y)

# Print Output
print 'Analytical Solution\n', analytical_solution(X,Y)
print 'Gradient Decent Solution\n', FinalTheta
print 'Iterations used = ', iteration

### 2D plot of the hypothesis function ###
X_plot = [item[1] for item in X.tolist()]
Y_plot = [item[0] for item in Y.tolist()]
x = np.arange(min(X_plot)-1, max(X_plot)+1, 0.1)

fig = plt.figure(figsize=(30, 30))

A = []; B = []; C = []
theta_0_plot = np.arange(-0.5, 2.5, 0.1)
theta_1_plot = np.arange(-1, 1, 0.1)
theta_0_plot, theta_1_plot = np.meshgrid(theta_0_plot, theta_1_plot)
Z = np.vectorize(create_J_plot)(theta_0_plot, theta_1_plot)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(theta_0_plot, theta_1_plot, Z, rstride=1, cstride=1, alpha=0.3, linewidth=0.1, cmap=cm.coolwarm)
ax3.set_zlim(0, 2)


# Plot
for index in range(iteration+1):
	line = Saved_Theta[index]
	# print line[0],line[1],line[2]
	plt.subplot(2, 2, 1)	
	plt.plot(X_plot, Y_plot, 'ro')
	ln, = plt.plot(x, linear2(line, x))
	plt.axis([min(X_plot)-0.1*np.std(X_plot), max(X_plot)+0.1*np.std(X_plot), min(Y_plot)-0.1*np.std(Y_plot), max(Y_plot)+0.1*np.std(Y_plot)])
	plt.ylabel('Density')
	plt.xlabel('Acidity')
	plt.title('Wine density with Acidity')

	plt.subplot(2, 2, 2)
	ln2, = plt.plot(range(1, index), Saved_J[1:index:1], '')
	plt.ylabel('Cost funtion')
	plt.xlabel('Iterations')
	plt.title('Cost with time')
	index = index + 1

	A.append(line[0]); B.append(line[1]); C.append(line[2])
	# wframe = ax3.plot_wireframe(A, B, C, rstride=1, cstride=1)
	point = ax3.plot([line[0]],[line[1]],[line[2]], 'r.')

	plt.subplot(2, 2, 4)
	CS = plt.contour(theta_0_plot, theta_1_plot, Z)
	plt.title('Contour Plot Showing Gradient Descent')
	point = plt.plot([line[0]],[line[1]], 'ro')
	point = plt.plot(A,B)

	plt.pause(0.0001)

	if index == iteration:
		break
	ln.remove()
	ln2.remove()

plt.show()