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
import time
from random import shuffle, random

# Variables used in learing algorithm
iteration = 0
LearningRate = 0.1
Epsilon = 1e-30
history = []
Saved_J = []

# Data variables
X_orig = []
Y = []
X = [] # Normalized X with X0 as 1

# Read Data from file and return numpy array
def read(filename):
	return np.matrix([map(float, line.strip().split()) for line in open(filename)])

# Return Hypothesis at X and Theta
def hypothesis(X, Theta):
	return X * Theta

# Return Hypothesis for a particular example X(i)
def hypothesis_particular(X, Theta, i):
	return X[i] * Theta

# Hypothesis function for a particular X
def hypothesis_x(Theta, x):
	return (Theta[1] * x) + Theta[0]

# Determine Cost function (J) for a given X and Y parameterized by Theta
def J(X, Y, Theta):
	return (np.linalg.norm(Y - hypothesis(X, Theta))**2 / (2*X.shape[0])).item(0)

# Determine gradient of cost function
def gradJ(X, Y, Theta):
	return X.T * (hypothesis(X, Theta) - Y) / X.shape[0]

# Determine gradient of cost function for a particular training example
def gradJ_sgd(X, Y, Theta, i):
	return X[i].T * (hypothesis_particular(X, Theta, i) - Y[i]) 

# Returns the value of J(theta)
def J_plot(Theta_0, Theta_1):
	Theta = np.matrix([[Theta_0],[Theta_1]])
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

# Returns normalised numpy array
def normalise(X):
    	return (X - np.mean(X, axis=0))/np.std(X, axis=0)

# Analytical Solution for linear regression 
def analytical_solution(X, Y):
    return np.linalg.inv(X.T * X) * X.T * Y

k = 0
# Batch Gradient Descent Algorithm
def gradient_descent(X, Y):
	global iteration
	global LearningRate
	global history 
	t = 0 # time variable
	Theta = np.matrix([[float(0)]] * (X.shape[1])) # Initialise theta based on input size
	j = J(X, Y, Theta)
	Saved_J.append(j)
	history.append([item for sublist in Theta.tolist() for item in sublist])
	while(True):
		# print LearningRate
		iteration += 1;
		start_time = time.time()
		grad = gradJ(X, Y, Theta)
		newTheta = Theta - LearningRate * grad
		newJ = J(X, Y, newTheta)
		Theta = newTheta
		t += (time.time() - start_time)
		if j - newJ < Epsilon: 	# Value has converged
			k+=1
			if k > 10:
				break
		else:
			k = 0
		j = newJ
		history.append([item for sublist in Theta.tolist() for item in sublist])
		Saved_J.append(j)
	print("Total time : "),
	print(t)
	return Theta

# Stochastic Gradient Descent Algorithm
def stochastic_gradient_descent(X, Y):
	global iteration
	global LearningRate
	global history
	t = 0 # time variable
	Theta = np.matrix([[float(0)]] * (X.shape[1])) # Initialise theta based on input size
	j = J(X, Y, Theta)
	Saved_J.append(j)
	history.append([item for sublist in Theta.tolist() for item in sublist])
	while(True):
		newTheta = Theta
		oldJ = j

		shuffled_list = range(1, X.shape[0])
		shuffle(shuffled_list) # Shuffle order of examples after each epoch
		for x in shuffled_list:
			start_time = time.time()
			grad = gradJ_sgd(X, Y, Theta, x)
			newTheta = Theta - LearningRate * grad
			newJ = J(X, Y, newTheta)
			Theta = newTheta
			t += (time.time() - start_time)
			history.append([item for sublist in Theta.tolist() for item in sublist])
			Saved_J.append(j)
			j = newJ
			iteration += 1;

		if oldJ - newJ < Epsilon: 	# Value has converged
			break
	print("Total time : "),
	print(t)			
	return Theta

# Read Data
X_orig = read('linearX.csv')
Y = read('linearY.csv')

# Prints size of training set
print ("Number of examples : %s" % X_orig.shape[0])

# Take learning rate from user
print("Enter Learning rate : "),
LearningRate = input()

# Take which Gradient Descent Algorithm to use from user
print("Enter 0 for BGD and 1 for SGD : "),
option = input()

# Normalize X values
X = np.c_[np.ones((X_orig.shape[0], 1)), normalise(X_orig)]

# Perform Gradient Descent
FinalTheta = []
if option == 1:
	FinalTheta = stochastic_gradient_descent(X, Y)
else:
	FinalTheta = gradient_descent(X, Y)

# Print results
print 'Iterations used = ', iteration
print 'Analytical Solution\n', analytical_solution(X,Y)
print 'Gradient Decent Solution\n', FinalTheta

# Create plots
Y_plot = [item[0] for item in Y.tolist()]
x = np.array([item[1] for item in X.tolist()])

fig = plt.figure(figsize=(30, 30))
fig.suptitle('Linear Regression')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

A = []; B = [];
theta0 = np.linspace(-0.25, 2, 100)
theta1 = np.linspace(-1, 1, 100)
theta0, theta1 = np.meshgrid(theta0, theta1)
Z = np.vectorize(J_plot)(theta0, theta1)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(theta0, theta1, Z, rstride=1, cstride=1, alpha=0.3, linewidth=0.1, cmap=cm.coolwarm)
ax3.set_zlim(min(Saved_J), max(Saved_J))

# Plot for each iteration
for index in range(iteration+1):
	line = history[index]
	plt.subplot(2, 2, 1)	
	plt.plot(X_orig, Y_plot, 'ro')
	ln, = plt.plot(X_orig, hypothesis_x(line, x))
	plt.axis([min(X_orig)-0.1*np.std(X_orig), max(X_orig)+0.1*np.std(X_orig), min(Y_plot)-0.1*np.std(Y_plot), max(Y_plot)+0.1*np.std(Y_plot)])
	plt.ylabel('Density')
	plt.xlabel('Acidity')
	plt.title('Wine density with Acidity')

	plt.subplot(2, 2, 2)
	ln2, = plt.plot(range(1, index), Saved_J[1:index:1], '')
	plt.ylabel('Cost funtion')
	plt.xlabel('Iterations')
	plt.title('Cost with time')
	index = index + 1

	A.append(line[0]); B.append(line[1]);
	wireframe, = ax3.plot(A, B, Saved_J[0:index:1])
	point = ax3.plot([line[0]],[line[1]],[Saved_J[index-1]], 'r.')

	plt.subplot(2, 2, 4)
	CS = plt.contour(theta0, theta1, Z)
	plt.title('Contour Plot Showing Gradient Descent')
	point = plt.plot([line[0]],[line[1]], 'ro')
	point = plt.plot(A,B)

	plt.pause(0.0001)

	if index == iteration-1:
		break
	ln.remove()
	ln2.remove()
	wireframe.remove()

gd = ''
if option == 0:
	gd = "BGD"
else:
	gd = "SGD"

plt.savefig('Linear-regression-'+gd+'-'+str(LearningRate)+'.png')
plt.show()