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
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, title='Confusion matrix',
           ylabel='True label', xlabel='Predicted label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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
    return w, b, alpha

def test(w, b, d, e, filename, alpha, Y, kernel_type):
    X1, Y1 = parseData(filename)
    X1, Y1 = convertLinear(X1, Y1, d, e, False)
    correct = 0
    total = 0
    for i in xrange(Y1.shape[0]):
        val = float(w*(X1[i].reshape(1, X.shape[1]).T)) + b if kernel_type == "linear" else np.sum(np.multiply(alpha, np.multiply(Y, gaussian_kernel(X1[i].reshape(1, X.shape[1]), X, 0.05))))
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
    confidence = np.zeros(10)
    for i in range(10):
        for j in range(i+1, 10):
            predicted_label,a,conf = svm_predict([1], x, models[i][j], "-q")
            clsfy = j if predicted_label[0] >= 0 else i
            wins[clsfy] += 1
            confidence[clsfy] += conf
    maxes = np.argwhere(wins == np.amax(wins))
    if maxes.size == 1:
        argmax = maxes[0][0]
    else:
        argmax = np.argwhere(confidence == np.amax(condifidence[maxes]))[0][0]
    return argmax, wins

def test_multiclass(models, X1, Y1):
    test_data, test_labels = listify(X1, Y1)
    correct = 0
    total = 0
    predicted = []
    for i in xrange(Y1.shape[0]):
        clsfy, wins = classify(models, [test_data[i]])
        predicted.append(clsfy)
        if clsfy == test_labels[i]:
            correct += 1
        else:
            print correct, total, wins, clsfy, test_labels[i]
            savefig(X1[i], "./wrong/wrong"+str(total)+"a"+str(int(test_labels[i]))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total), predicted, test_labels

trainfile = "train.csv"
testfile = "test2.csv"

# Read data from file
X, Y = parseData(trainfile)
print("Data parse complete...")

d = 0

print "D = ", d
print "ConvOpt results:"

Xd, Yd = convertLinear(X, Y, d, (d+1)%10)

# Linear SVM Model
w, b, a = train(Xd, Yd, "linear", 1, 0)
print "Accuracy (Linear Kernel) = ", test(w, b, d, (d+1)%10, testfile, X, Y, "linear")*100

# Gaussian SVM Model
# w, b, a = train(Xd, Yd, "gaussian", 1, 0.05)
# print "Accuracy (Gaussian Kernel) = ", test(w, b, d, (d+1)%10, testfile, X, a, "gaussian")*100

print "LibSVM results:"

train_data, train_labels = listify(Xd, Yd)

Xt, Yt = parseData(testfile)
X1, Y1 = convertLinear(Xt, Yt, d, (d+1)%10, True)
test_data, test_labels = listify(X1, Y1)

# Linear SVM Model
model = svm_train(train_labels, train_data,'-q -t 0 -c 1')
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model, "-q")
print "Accuracy (Linear Kernel) = ", accuracy[0]

# Gaussian SVM Model
model = svm_train(train_labels, train_data,'-q -g 0.05 -c 1')
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model, "-q")
print "Accuracy (Gaussian Kernel) = ", accuracy[0]

# Multiclass model

Xtest, Ytest = parseData(testfile)

# Linear SVM Model
# models = train_multiclass(X, Y, '-t 0 -c 1 -q')
# acc = test_multiclass(models, Xtest, Ytest)

# print "Multiclass Accuracy (Linear Kernel) = ", acc*100

# Gaussian SVM Model
models = train_multiclass(X, Y, '-g 0.05 -c 1 -q')
acc, pred, actual = test_multiclass(models, Xtest, Ytest)

print "Multiclass Accuracy (Gussian Kernel) = ", acc*100

cm = confusion_matrix(actual, pred)

print(cm)

plot_confusion_matrix(cm, [1, 2, 3, 4, 5])
plt.savefig("Confusion-Matrix")

# Validation

Xv = X[0:X.size[0]/10:1]
Yv = Y[0:Y.size[0]/10:1]

Xtrain = X[X.size[0]/10::1]
Ytrain = Y[Y.size[0]/10::1]

for i in [0.00001, 0.001, 1, 5, 10]:
    models = train_multiclass(Xtrain, Ytrain, '-g 0.05 -c '+str(i)+' -q')
    acc, pred, actual = test_multiclass(models, Xv, Yv)
    print "Validation Accuracy with C = ", i, " is : ", acc*100
    acc, pred, actual = test_multiclass(models, Xtest, Ytest)
    print "Test Accuracy with C = ", i, " is : ", acc*100