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

def convertLinear(X, Y, d, e, changeY=True):
    retY = Y[np.logical_or.reduce([Y == x for x in [d,e]])]
    if changeY:
        retY[retY == d] = -1
        retY[retY == e] = 1
    retX = np.array(X[np.where(np.logical_or.reduce([Y == x for x in [d,e]]))[0], :])
    return retX, retY.T

def savefig(x, name="Image"):
    plt.imshow(x.reshape(28,28), cmap='gist_gray', interpolation='nearest')
    plt.savefig(name)

def norm(x, y):
    return np.linalg.norm(x - y)

def linear_kernel(X, Y):
	A = np.multiply(X, Y)
	return A*A.T

def gaussian_kernel(X, Y, gamma):
	M = [map(float, [0]) * Y.shape[0] for _ in xrange(Y.shape[0])]
	for i in xrange(Y.shape[0]):
		for j in xrange(Y.shape[0]):
			M[i][j] = Y.item(i)*Y.item(j)*(math.exp(-gamma * np.square(norm(X[i], X[j])).item()))
	return M

def weight_vector(X, Y, a):
    return np.sum(np.multiply(X, np.multiply(Y, a)), axis = 0)

def intercept_l(X, Y, w):
    temp = w*X.T
    return -0.5*(np.max(np.matrix(temp[Y.T == -1])) + np.min(np.matrix(temp[Y.T == 1])))

def intercept_g(X, Y, a):
    	m = Y.shape[0]
	mat = np.matrix([map(float, [0]) for _ in xrange(m)])
	for i in xrange(m):
		if(a[i].value > 499.9 or a[i].value < 0.1):
			continue
		temp = 0
		for j in xrange(m):
			temp += a[j].value*Y.item(j)*gaussian(X[i], X[j])
		mat[i] = temp
	maxmat = np.matrix(mat[Y > 0])
	minmat = np.matrix(mat[Y < 0])
	maxm = np.matrix(maxmat[maxmat != 0])
	minm = np.matrix(minmat[minmat != 0])
	max = np.max(minm)
	min = np.min(maxm)
	return -0.5*(max + min)


def train(X, Y, kernel_type, C=1, gamma=0.05):
    alpha = cvx.Variable((Y.shape[0], 1)) # Variable for optimization
    Q = linear_kernel(X, Y) if kernel_type == "linear" else gaussian_kernel(X, Y, gamma) # Kernel
    objective = cvx.Maximize(cvx.sum(alpha) - 0.5*(cvx.quad_form(alpha, Q))) # Objective funtion
    constraints = [alpha >= 0, alpha <= C, alpha.T*Y == 0] # Constraints
        
    cvx.Problem(objective, constraints).solve() 
    print alpha.value
    index = np.zeros((alpha.value.size, 1)) # indentify support vectors
    for i in xrange(alpha.size):
        index[i,0] = alpha[i].value
        if alpha[i].value > 0.1:
            print i
            savefig(X[i].reshape(1, 784), "./sv/supportvector"+str(i)+"y"+str(Y[i])+".png")
        
    w = weight_vector(X, Y, index)
    b = intercept_l(X, Y, w) if kernel_type == "linear" else intercept_g(X, Y, alpha)
    return w, b

def test(w, b, filename):
    X1, Y1 = parseData(filename)
    X1, Y1 = convertLinear(X1, Y1, d, False)
    correct = 0
    total = 0
    for i in xrange(Y1.shape[0]):
        val = float(w*(X1[i].reshape(1, X.shape[1]).T)) + b
        clsfy = d+1 if val >= 0 else d
        if clsfy == Y1.item(i):
            correct += 1
        else:
            savefig(X1[i].reshape(1, X.shape[1]), "./wrong/wrong"+str(total)+"a"+str(int(Y1.item(i)))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total)

def train_multiclass(X, Y, kernel_type, C=1, gamma=0.05):
    w = np.empty((10, 10))
    b = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            if i != j:
                Xd, Yd = convertLinear(X, Y, i, j)
                w[i][j], b[i][j] = train(X, Y, kernel_type, X, gamma)
    return w, b

def classify(w, b, x):
    wins = np.zeros(10)
    for i in range(10):
        for j in range(10):
            if i != j:
                val = float(w*(x.reshape(1, 784).T)) + b
                clsfy = j if val >= 0 else x
                wins[clsdfy] += 1
    return wins.argmax() + 1

def test_multiclass(w, b, filename):
    X1, Y1 = parseData(filename)
    correct = 0
    total = 0
    for i in xrange(Y1.shape[0]):
        clsfy = classify(w, b, X1[i])
        if clsfy == Y1.item(i):
            correct += 1
        else:
            savefig(X1[i].reshape(1, X.shape[1]), "./wrong/wrong"+str(total)+"a"+str(int(Y1.item(i)))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total)


# Read data from file
X, Y = parseData("train.csv")

d = 0

Xd, Yd = convertLinear(X, Y, d, (d+1)%10)

# w, b = train(Xd, Yd, "linear", 1, 0)

# print "accuracy (Linear Kernel) = ", test(w, b, "test.csv")

w, b = train(Xd, Yd, "gaussian", 1, 0.005)

print "accuracy (Linear Kernel) = ", test(w, b, "test.csv")