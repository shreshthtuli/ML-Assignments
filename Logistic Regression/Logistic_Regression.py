"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Logistic Regression

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
LearningRate = 0.5
Epsilon = 1e-10
history = []
Saved_L = []

# Data variables
X_orig = []
Y = []
X = [] # Normalized X with X0 as 1

# Read Data from file and return numpy array
def read(filename):
	return np.matrix([map(float, line.strip().split(',')) for line in open(filename)])

# Return Hypothesis at X and Theta
def hypothesis(X, Theta, i):
	return 1 / (1 + math.exp(-X[i] * Theta))

# Hypothesis function for a particular x2 and x1
def hypothesis_x(Theta, x1, x2):
	return 1 / (1 + math.exp((Theta[2] * x2) + (Theta[1] * x1) + Theta0))

# Determine Likelihood function (L) for a given X and Y parameterized by Theta
def L(X, Y, Theta):
	Likelihood = np.matrix([[hypothesis(X, Theta, i)**Y[i].item() + (1-hypothesis(X, Theta, i))**(1-Y[i].item())] for i in range(X.shape[0])])
	return np.sum(Likelihood)

# Determine gradient of cost function
def gradJ(X, Y, Theta):
	hypo = np.matrix([[hypothesis(X, Theta, i)] for i in range(X.shape[0])])
	return X.T * (Y - hypo) / X.shape[0]

# Determine gradient of cost function for a particular training example
def gradJ_sgd(X, Y, Theta, i):
	return X[i].T * (Y[i] - hypothesis(X, Theta, i)) 

# Returns normalised numpy array
def normalise(X):
	return (X - np.mean(X, axis=0))/np.std(X, axis=0)

def boundary_line(x, Theta):
	return (-Theta.item(1)*x-Theta.item(0))/Theta.item(2)

# Hessian
def Hessian(X, Y, Theta):
	temp = [hypothesis(X, Theta, i) * (1 - hypothesis(X, Theta, i)) for i in range(X.shape[0])]
	D = np.diag(temp)
	return -(X.T * D * X) / X.shape[0]

# Batch Gradient Descent Algorithm
def gradient_descent(X, Y):
	global iteration
	global LearningRate
	global history 
	t = 0 # time variable
	Theta = np.matrix([[float(random())]] * (X.shape[1])) # Initialise theta based on input size
	l = L(X, Y, Theta)
	Saved_L.append(l)
	history.append([item for sublist in Theta.tolist() for item in sublist])
	while(True):
		iteration += 1
		# print Theta
		start_time = time.time()
		grad = gradJ(X, Y, Theta)
		newTheta = Theta + LearningRate * grad
		newL = L(X, Y, newTheta)
		t += (time.time() - start_time)
		print math.fabs(np.linalg.norm(Theta) - np.linalg.norm(newTheta))
		if math.fabs(np.linalg.norm(Theta) - np.linalg.norm(newTheta)) < 10*Epsilon: 	# Value has converged
			break	
		Theta = newTheta		
		l = newL
		history.append([item for sublist in Theta.tolist() for item in sublist])
		Saved_L.append(l)
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
	l = L(X, Y, Theta)
	Saved_L.append(l)
	history.append([item for sublist in Theta.tolist() for item in sublist])
	while(True):
		newTheta = Theta
		oldL = l

		shuffled_list = range(1, X.shape[0])
		shuffle(shuffled_list) # Shuffle order of examples after each epoch
		for x in shuffled_list:
			start_time = time.time()
			grad = gradJ_sgd(X, Y, Theta, x)
			newTheta = Theta + (0.5)*LearningRate * grad
			newL = L(X, Y, newTheta)
			Theta = newTheta
			t += (time.time() - start_time)
			history.append([item for sublist in Theta.tolist() for item in sublist])
			Saved_L.append(l)
			l = newL
			iteration += 1;

		print math.fabs(oldL - newL)
		if math.fabs(oldL - newL) < 0.001: 	# Value has converged
			break
	print("Total time : "),
	print(t)			
	return Theta

# Perform Fischer method i.e. Newton's method on Logistic Regression
def newtons_method(X, Y):
	global iteration
	global LearningRate
	global history 
	t = 0 # time variable
	Theta = np.matrix([[float(random())]] * (X.shape[1])) # Initialise theta based on input size
	l = L(X, Y, Theta)
	Saved_L.append(l)
	history.append([item for sublist in Theta.tolist() for item in sublist])
	while(True):
		iteration += 1
		# print Theta
		start_time = time.time()
		grad = gradJ(X, Y, Theta)
		newTheta = Theta - np.linalg.inv(Hessian(X, Y, Theta)) * gradJ(X, Y, Theta)
		newL = L(X, Y, newTheta)
		t += (time.time() - start_time)
		# print math.fabs(np.linalg.norm(Theta) - np.linalg.norm(newTheta))
		if math.fabs(np.linalg.norm(Theta) - np.linalg.norm(newTheta)) < Epsilon: 	# Value has converged
			break	
		Theta = newTheta		
		l = newL
		history.append([item for sublist in Theta.tolist() for item in sublist])
		Saved_L.append(l)
	print("Total time : "),
	print(t)
	return Theta

# Read Data
X_orig = read('logisticX.csv')
Y = read('logisticY.csv')

# Prints size of training set
print ("Number of examples : %s" % X_orig.shape[0])

# Take learning rate from user
print("Enter Learning rate : "),
LearningRate = input()

# Take which Gradient Descent Algorithm to use from user
print("Enter 0 for BGD and 1 for SGD and 2 for Newton's method: "),
option = input()

# Normalize X values
X = np.c_[np.ones((X_orig.shape[0], 1)), normalise(X_orig)]

# Perform Gradient Descent
FinalTheta = []
if option == 1:
	FinalTheta = stochastic_gradient_descent(X, Y)
elif option == 0:
	FinalTheta = gradient_descent(X, Y)
else:
    FinalTheta = newtons_method(X, Y)

# Print results
print 'Iterations used = ', iteration
print 'Final Solution\n', FinalTheta

X_one = []
X_zero = []
for i in range(len(Y.tolist())):
	if Y.item(i) > 0.5:
		X_one.append([X.tolist()[i][1], X.tolist()[i][2]])
	else:
		X_zero.append([X.tolist()[i][1], X.tolist()[i][2]])


# Boundary
x = np.arange(-2, 2.2, 0.1)
# plt.plot(x, boundary_line(x, FinalTheta))
fig = plt.figure(figsize=(30, 30))
plt.legend()
index = 0

while index < iteration:
	Theta = np.array(history[index])
	# Plot 
	plt.subplot(1, 2, 1)
	X_plot1 = [item[0] for item in X_one]
	Y_plot1 = [item[1] for item in X_one]
	plt.plot(X_plot1, Y_plot1, 'ro', label='Class 1')

	X_plot0 = [item[0] for item in X_zero]
	Y_plot0 = [item[1] for item in X_zero]
	plt.plot(X_plot0, Y_plot0, 'rx', label='Class 2')

	plt.axis([min(min(X_plot0),min(X_plot1))-0.2, max(max(X_plot0),max(X_plot1))+0.2, 
		min(min(Y_plot0),min(Y_plot1))-0.2, max(max(Y_plot0),max(Y_plot1))+0.2])
	ln, = plt.plot(x, boundary_line(x, Theta))
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.title('Boundary between the labelled examples')

	plt.subplot(1, 2, 2)
	ln2, = plt.plot(range(1, index), Saved_L[1:index:1], '')
	plt.ylabel('Likelihood function')
	plt.xlabel('Iterations')
	plt.title('Likelihood with time')

	plt.pause(0.1)
	if index == iteration-1:
		break
	elif index == 0:
		plt.legend()
	ln.remove()
	index = int(1.1*index) + 1

gd = ''
if option == 0:
	gd = "BGD"
elif option == 1:
	gd = "SGD"
else:
	gd = "Newton"

plt.savefig('Logistic-regression-'+gd+'-'+str(LearningRate)+'.png')