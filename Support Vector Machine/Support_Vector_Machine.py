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
from svmutil import *

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

def linear_kernel(X, Y):
	return np.multiply(X, Y)*np.multiply(X, Y).T

def gaussian_kernel(X, Y, gamma):
	M = [map(float, [0]) * Y.shape[0] for _ in xrange(Y.shape[0])]
	for i in xrange(Y.shape[0]):
		for j in xrange(Y.shape[0]):
			M[i][j] = Y.item(i)*Y.item(j)*(math.exp(-gamma * np.square(np.linalg.norm(X[i] - X[j])).item()))
	return M

def weight_vector(X, Y, a):
    return np.sum(np.multiply(X, np.multiply(Y, a)), axis = 0)

def intercept(x, y, w):
	return y - w*(x.reshape(1,784).T)

def listify(X, Y):
    retx = []
    rety = []
    for i in xrange(Y.shape[0]):
        rety.append(int(Y.item(i)))
    for i in xrange(X.shape[0]):
        param = []
        for j in xrange(X.shape[1]):
            param.append(X.item(i,j))
        retx.append(param)
    return retx, rety

def train(X, Y, kernel_type, C=1, gamma=0.05):
    alpha = cvx.Variable((Y.shape[0], 1)) # Variable for optimization
    Q = linear_kernel(X, Y) if kernel_type == "linear" else gaussian_kernel(X, Y, gamma) # Kernel
    objective = cvx.Maximize(cvx.sum(alpha) - 0.5*(cvx.quad_form(alpha, Q))) # Objective funtion
    constraints = [alpha >= 0, alpha <= C, alpha.T*Y == 0] # Constraints
        
    cvx.Problem(objective, constraints).solve() 
    # print alpha.value
    index = np.zeros((alpha.value.size, 1)) # indentify support vectors
    sv = 0; count = 0
    for i in xrange(alpha.size):
        index[i,0] = alpha[i].value
        if alpha[i].value > 0.1 and alpha[i].value <= 1:
            # print i
            sv = i; count += 1
            # savefig(X[i].reshape(1, 784), "./sv/supportvector"+str(i)+"y"+str(Y[i])+".png")
    print "Num support vectors : ", count
    w = weight_vector(X, Y, index)
    b = intercept(X[sv], Y[sv], w)
    return w, b

def test(w, b, d, e, filename):
    X1, Y1 = parseData(filename)
    X1, Y1 = convertLinear(X1, Y1, d, e, False)
    correct = 0
    total = 0
    for i in xrange(Y1.shape[0]):
        val = float(w*(X1[i].reshape(1, X.shape[1]).T)) + b
        clsfy = e if val >= 0 else d
        if clsfy == Y1.item(i):
            correct += 1
        # else:
        #     savefig(X1[i].reshape(1, X.shape[1]), "./wrong/wrong"+str(total)+"a"+str(int(Y1.item(i)))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total)

def train_multiclass(X, Y, param):
    models = np.empty((10, 10),dtype=object)
    for i in range(10):
        for j in range(i+1,10):
            Xd, Yd = convertLinear(X, Y, i, j, True)
            train_data, train_labels = listify(Xd, Yd)
            m = svm_train(train_labels, train_data, param)
            models[i][j] = m
    return models

def classify(models, x):
    wins = np.zeros(10)
    for i in range(10):
        for j in range(i+1, 10):
            predicted_label,a,b = svm_predict([1], x, models[i][j], "-q")
            clsfy = j if predicted_label[0] >= 0 else i
            wins[clsfy] += 1
    return wins.argmax()

def test_multiclass(models, filename):
    X1, Y1 = parseData(filename)
    test_data, test_labels = listify(X1, Y1)
    correct = 0
    total = 0
    for i in xrange(Y1.shape[0]):
        clsfy = classify(models, [test_data[i]])
        if clsfy == test_labels[i]:
            correct += 1
        else:
            savefig(X1[i], "./wrong/wrong"+str(total)+"a"+str(int(test_labels[i]))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total)


# Read data from file
X, Y = parseData("train2.csv")
print("Data parse complete...")

d = 0

print "D = ", d
print "ConvOpt results:"

Xd, Yd = convertLinear(X, Y, d, (d+1)%10)

# Linear SVM Model
w, b = train(Xd, Yd, "linear", 1, 0)
print "Accuracy (Linear Kernel) = ", test(w, b, d, (d+1)%10, "test2.csv")

# Gaussian SVM Model
w, b = train(Xd, Yd, "gaussian", 1, 0.05)
print "Accuracy (Gaussian Kernel) = ", test(w, b, d, (d+1)%10, "test2.csv")

print "LibSVM results:"

train_data, train_labels = listify(Xd, Yd)

X1, Y1 = convertLinear(X, Y, d, (d+1)%10, True)
test_data, test_labels = listify(X1, Y1)

# Linear SVM Model
model = svm_train(train_labels, train_data,'-q -t 0 -c 1')
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model. "-q")
print "Accuracy (Linear Kernel) = ", accuracy

# Gaussian SVM Model
model = svm_train(train_labels, train_data,'-q -g 0.05 -c 1')
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model, "-q")
print "Accuracy (Gaussian Kernel) = ", accuracy

# Multiclass model

# Linear SVM Model
models = train_multiclass(X, Y, '-t 0 -c 1 -q')
acc = test_multiclass(models, "test2.csv")

print "Multiclass Accuracy (Linear Kernel) = ", acc

# Gaussian SVM Model
models = train_multiclass(X, Y, '-g 0.05 -c 1 -q')
acc = test_multiclass(models, "test2.csv")

print "Multiclass Accuracy (Gussian Kernel) = ", acc