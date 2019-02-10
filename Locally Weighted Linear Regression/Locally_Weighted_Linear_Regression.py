"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Locally Weighted Linear Regression

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
Tau = 1

# Data variables
X_orig = []
Y = []
X = [] # Normalized X with X0 as 1
W = []

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
def J(X, Y, W, Theta):
	return (Y - hypothesis(X, Theta)).T * W * (Y - hypothesis(X, Theta)) / (2*X.shape[0]).item(0)

# Determine gradient of cost function
def gradJ(X, Y, W, Theta):
	return X.T * W * (hypothesis(X, Theta) - Y) / X.shape[0]

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

def get_W(x, X, Tau):
	W = []
	for i in range(X.shape[0]):
		rowlist = []
		for j in range(X.shape[0]):
			if(i == j):
 				rowlist.append(math.exp(-(x - X[i].tolist()[0][1])**2 / (2 * Tau**2)))
			else:
				rowlist.append(0)
		W.append(rowlist)
	return np.matrix(W)

# Analytical Solution for linear regression 
def analytical_solution(X, Y):
    return np.linalg.inv(X.T * X) * X.T * Y

# Analytical Solution for locally weighted linear regression 
def weighted_analytical_solution(x, X, Y):
	W = get_W(x, X, Tau)
	return np.linalg.pinv(X.T * W * X) * X.T * W * Y

def give_y(x, Theta):
     	return Theta.item(0) + x*Theta.item(1)

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
			break		
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
X_orig = read('weightedX.csv')
Y = read('weightedY.csv')

# Prints size of training set
print ("Number of examples : %s" % X_orig.shape[0])

# Take learning rate from user
print("Enter Learning rate : "),
LearningRate = input()

# Take bandwidth parameter from user
print("Enter bandwidth parameter : "),
Tau = input()

# Take which Gradient Descent Algorithm to use from user
print("Enter 0 for BGD and 1 for SGD and 2 for analytical: "),
option = input()

option = 2

# Normalize X values
X = np.c_[np.ones((X_orig.shape[0], 1)), normalise(X_orig)]

# Perform Gradient Descent
FinalTheta = []
if option == 1:
	FinalTheta = stochastic_gradient_descent(X, Y)
elif option == 0:
	FinalTheta = gradient_descent(X, Y)

# Print results
print 'Iterations used = ', iteration

# Create plots
Y_plot = [item[0] for item in Y.tolist()]
X_plot = [item[1] for item in X.tolist()]
x = np.array([item[1] for item in X.tolist()])

fig = plt.figure(figsize=(30, 30))

xspace = np.linspace(np.min(X_plot), np.max(X_plot), 1000).tolist()

# Theta aray
All_Thetas = []

# yspace
yspace = []

for x in xspace:
	yspace.append(give_y(x, weighted_analytical_solution(x, X, Y)))

plt.subplot(1, 2, 1)
plt.plot(X_plot, Y_plot, 'ro')
# Plot linear regression analytical solution
plt.subplot(1, 2, 1)	
plt.plot(X_plot, Y_plot, 'ro')
ln, = plt.plot(X_plot, hypothesis_x(analytical_solution(X, Y), np.array(X_plot)).T)
plt.axis([min(X_plot)-0.1*np.std(X_plot), max(X_plot)+0.1*np.std(X_plot), min(Y_plot)-0.1*np.std(Y_plot), max(Y_plot)+0.1*np.std(Y_plot)])
plt.title("Linear Regression")

# Plot weighted linear regression analytical solution
plt.subplot(1, 2, 2)
plt.plot(X_plot, Y_plot, 'ro')

plt.plot(xspace, yspace)
plt.ylabel("Y")
plt.xlabel("X")
title = "Locally weighted linear regression for Tau = " + str(Tau)
plt.title(title)
plt.suptitle("Locally Weighted Linear Regression")
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

# for i in range(1, 100, 10):
# 	# Plot weighted linear regression analytical solution
# 	plt.subplot(1, 2, 2)	
# 	Tau = 10/float(i)
# 	yspace = []
# 	for x in xspace:
# 		yspace.append(give_y(x, weighted_analytical_solution(x, X, Y)))
# 	a, = plt.plot(xspace, yspace)
# 	title = "Locally weighted linear regression for Tau = " + str(Tau)
# 	plt.title(title)
# 	plt.pause(0.0001)
# 	if i == 9:
# 		break;
# 	a.remove()

gd = ''
if option == 0:
	gd = "BGD"
elif option == 1:
	gd = "SGD"
else:
    gd = "Analytical"


plt.savefig('Weighted-Linear-regression-'+gd+'-'+str(LearningRate)+'-'+str(Tau)+'.png')
plt.show()