"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Support Vector Machine

"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import math
# import svmutil as svm

def parseData(filename):
    read = np.matrix([map(float, line.strip().split(',')) for line in open(filename)])
    X = read[:,0:-1]/255
    Y = read[:,-1]
    return X, Y

def convertLinear(d):
    retY = Y[np.logical_or.reduce([Y == x for x in [d,d+1]])]
    retY[retY == d] = -1
    retY[retY == d+1] = 1
    retX = np.array(X[np.where(np.logical_or.reduce([Y == x for x in [d,d+1]]))[0], :])
    return retX, retY.T

def savefig(x, name="Image"):
    plt.imshow(x.reshape(28,28), cmap='grey', interpolation='nearest')
    plt.savefig(name)

def linear_kernel(X, Y):
	A = np.multiply(X, Y)
	return A*A.T

def gaussian_kernel(X, Y, gamma):
	M = [map(float, [0]) * m for _ in xrange(Y.shape[0])]
	for i in xrange(m):
		for j in xrange(m):
			M[i][j] = Y.item(i)*Y.item(j)*gaussian(math.exp(-gamma * np.square(norm(X[i], X[j])).item()))
	return M

def weight_vector(X, Y, a):
    mat = np.multiply(Y, a)
    return np.sum(np.multiply(X, mat), axis = 0)

def intercept(X, Y, w):
    mat = w*X.T
    return -0.5*(np.max(np.matrix(mat[Y.T == -1])) + np.min(np.matrix(mat[Y.T == 1])))

def train(X, Y, kernel_type, C=1, gamma=0.05):
    alpha = cvx.Variable((Y.shape[0], 1)) # Variable for optimization
    Q = linear_kernel(X, Y) if kernel_type == "linear" else gaussian_kernel(X, Y, gamma) # Kernel
    objective = cvx.Maximize(cvx.sum(alpha) - 0.5*(cvx.quad_form(alpha, Q))) # Objective funtion
    constraints = [alpha >= 0, alpha <= C, (alpha.T*Y) == 0] # Constraints
        
    cvx.Problem(objective, constraints).solve() 
    return alpha


# Read data from file
X, Y = parseData("train.csv")

d = 0

Xd, Yd = convertLinear(d)

alpha = train(Xd, Yd, "linear", 1, 0)
print alpha.value
index = np.zeros((alpha.value.size, 1)) # indentify support vectors
for i in xrange(alpha.size):
	index[i,0] = alpha[i].value
	if alpha[i].value > 0.1 and alpha[i].value < 499.9:
		print i

# Calculate weight vector
w = weight_vector(Xd, Yd, index)

# Calculate intercept b
b = intercept(Xd, Yd, w)

# Test on test data
X1, Y1 = parseData("test.csv")
correct = 0
count = 0
for i in xrange(Y1.shape[0]):
    val = float(w*X1[i].T) + b
    print val
    if val >= 0:
        clsfy = d+1
    else:
        clsfy = d
    if clsfy == Y1.item(i):
        correct += 1
    count += 1

print "accuracy (Linear Kernel) = ", float(correct) / float(count)
