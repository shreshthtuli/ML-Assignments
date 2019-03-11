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
import time
from sys import argv

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

def gaussian_weight(X, Ya, Xn):
    K = np.array(gaussian_kernel(Xn, X, 0.05))
    print K.shape, Ya.T.shape
    res = K.dot(Ya.T)
    return np.sum(res)/res.size

def intercept(x, y, w):
	return (y - w*(x.reshape(1,784).T)).item(0)

def gaussian_intercept(X, Y, alpha):
    K = np.array(gaussian_kernel(X, X, 0.05))
    a = np.multiply(Y, alpha.value)
    res = (Y - a.T.dot(K).transpose())
    res = np.sum(res)/res.size
    return res

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
    
    w = weight_vector(X, Y, index)
    b = intercept(X[sv], Y[sv], w) if kernel_type == "linear" else gaussian_intercept(X, Y, alpha)
    return w, b, alpha, count

def test(w, b, d, e, filename, alpha, X, Y, kernel_type):
    X1, Y1 = parseData(filename)
    X1, Y1 = convertLinear(X1, Y1, d, e, False)
    correct = 0
    total = 0
    Ya = np.multiply(Y, alpha.value).T if kernel_type != "linear" else []
    for i in xrange(Y1.shape[0]):
        val = float(w*(X1[i].reshape(1, X.shape[1]).T)) + b# if kernel_type == "linear" else gaussian_weight(X, Ya, X1[i]) + b
        clsfy = e if val >= 0 else d
        if clsfy == Y1.item(i):
            correct += 1
        else:
            savefig(X1[i].reshape(1, X.shape[1]), "./wrong/wrong"+str(total)+"a"+str(int(Y1.item(i)))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total)

def train_multiclass_cvx(X, Y, kernel):
    w = np.empty((10, 10),dtype=object)
    b = np.empty((10, 10))
    for i in range(10):
        for j in range(i+1,10):
            Xd, Yd = convertLinear(X, Y, i, j, True)
            w[i][j], b[i][j], a, c = train(Xd, Yd, kernel, 1, 0.05)
    return w, b

def classify_cvx(w, b, x):
    wins = np.zeros(10)
    confidence = np.zeros(10)
    for i in range(10):
        for j in range(i+1, 10):
            val = float(w[i][j]*(x.T)) + b[i][j]
            clsfy = j if val >= 0 else i
            wins[clsfy] += 1
            confidence[clsfy] += math.fabs(val)
    maxes = np.argwhere(wins == np.amax(wins))
    if maxes.size == 1:
        argmax = maxes[0][0]
    else:
        argmax = np.argwhere(confidence == np.amax(confidence[maxes]))[0][0]
    return argmax, wins

def test_multiclass_cvx(w, b, X1, Y1):
    correct = 0
    total = 0
    predicted = []
    actual = []
    for i in xrange(Y1.shape[0]):
        clsfy, wins = classify_cvx(w, b, X1[i])
        predicted.append(clsfy); actual.append(Y1.item(i))
        if clsfy == Y1.item(i):
            correct += 1
        # else:
        #     print correct, total, wins, clsfy, test_labels[i]
        #     savefig(X1[i], "./wrong/wrong"+str(total)+"a"+str(int(test_labels[i]))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total), predicted, test_labels

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
            confidence[clsfy] += math.fabs(conf[0][0])
    maxes = np.argwhere(wins == np.amax(wins))
    if maxes.size == 1:
        argmax = maxes[0][0]
    else:
        argmax = np.argwhere(confidence == np.amax(confidence[maxes]))[0][0]
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
        # else:
        #     print correct, total, wins, clsfy, test_labels[i]
        #     savefig(X1[i], "./wrong/wrong"+str(total)+"a"+str(int(test_labels[i]))+"p"+str(int(clsfy))+".png")
        total += 1
    
    return float(correct) / float(total), predicted, test_labels

trainfile = argv[1]
testfile = argv[2]

# Read data from file
X, Y = parseData(trainfile)
print("Data parse complete...")

bin_or_mult = int(argv[3])

d = 0

print "D = ", d

print '\033[95m'+"---Binary Classification---"+'\033[0m'

########## BINARY CONVOPT ##########

print
print '\033[94m'+"ConvOpt results:"+'\033[0m'

Xd, Yd = convertLinear(X, Y, d, (d+1)%10)

# Linear SVM Model
start = time.time()
w, b, a, n = train(Xd, Yd, "linear", 1, 0)
end = time.time() - start
print "Accuracy (Linear Kernel) = ", test(w, b, d, (d+1)%10, testfile, a, Xd, Yd, "linear")*100
# print "Weight ", w
print "Bias ", b
print "nSV ", n
print "Time ", end

# Gaussian SVM Model
start = time.time()
w, b, a, n = train(Xd, Yd, "gaussian", 1, 0.05)
end = time.time() - start
print "Accuracy (Gaussian Kernel) = ", test(w, b, d, (d+1)%10, testfile, a, Xd, Yd, "gaussian")*100
# print "Weight ", w
print "Bias ", b
print "nSV ", n
print "Time ", end


########## BINARY LIBSVM ##########

print
print '\033[94m'+"LibSVM results:"+'\033[0m'

train_data, train_labels = listify(Xd, Yd)

Xt, Yt = parseData(testfile)
X1, Y1 = convertLinear(Xt, Yt, d, (d+1)%10, True)
test_data, test_labels = listify(X1, Y1)

# Linear SVM Model
start = time.time()
model = svm_train(train_labels, train_data,'-t 0 -c 1')
end = time.time() - start
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model, "-q")
print "Accuracy (Linear Kernel) = ", accuracy[0]
print "Time ", end
# print "Weight ", w

# Gaussian SVM Model
start = time.time()
model = svm_train(train_labels, train_data,'-g 0.05 -c 1')
end = time.time() - start
[predicted_label, accuracy, decision_values] = svm_predict(test_labels, test_data, model, "-q")
print "Accuracy (Gaussian Kernel) = ", accuracy[0]
print "Time ", end

########## MULTICLASS CONVOPT ##########

print '\033[95m'+"---Multiclass Classification---"+'\033[0m'

# Test data
Xtest, Ytest = parseData(testfile)
# Training accuracy

print
print '\033[94m'+"ConvOpt results:"+'\033[0m'

# Linear SVM Model
start = time.time()
w, b = train_multiclass_cvx(X, Y, 'linear')
end = time.time() - start
acc, pred, actual = test_multiclass_cvx(w, b, X, Y)
acc1, pred1, actual1 = test_multiclass_cvx(w, b, Xtest, Ytest)
print "Multiclass Training Accuracy (Linear Kernel) = ", acc*100
print "Multiclass Test Accuracy (Linear Kernel) = ", acc1*100
print "Time ", end

# Gaussian SVM Model
start = time.time()
w, b = train_multiclass_cvx(X, Y, 'gaussian')
end = time.time() - start
acc, pred, actual = test_multiclass_cvx(w, b, X, Y)
acc1, pred1, actual1 = test_multiclass_cvx(w, b, Xtest, Ytest)
print "Multiclass Training Accuracy (Gaussian Kernel) = ", acc*100
print "Multiclass Test Accuracy (Gaussian Kernel) = ", acc1*100
print "Time ", end


########## MULTICLASS LIBSVM ##########
print
print '\033[94m'+"LibSVM results:"+'\033[0m'

# Linear SVM Model
start = time.time()
models = train_multiclass(X, Y, '-t 0 -c 1 -q')
end = time.time() - start
acc, pred, actual = test_multiclass(models, X, Y)
acc1, pred1, actual1 = test_multiclass(models, Xtest, Ytest)
print "Multiclass Training Accuracy (Linear Kernel) = ", acc*100
print "Multiclass Test Accuracy (Linear Kernel) = ", acc1*100
print "Time ", end

# Gaussian SVM Model
start = time.time()
models = train_multiclass(X, Y, '-g 0.05 -c 1 -q')
end = time.time() - start
acc, pred, actual = test_multiclass(models, X, Y)
acc1, pred1, actual1 = test_multiclass(models, Xtest, Ytest)
print "Multiclass Training Accuracy (Gaussian Kernel) = ", acc*100
print "Multiclass Test Accuracy (Gaussian Kernel) = ", acc1*100
print "Time ", end

########## CONFUSION MATRIX ##########

cm = confusion_matrix(actual1, pred1)

print '\033[94m'+"Confusion Matrix:"+'\033[0m'
print(cm)

plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.savefig("Confusion-Matrix")


########## VALIDATION ##########

Xv = X[0:X.shape[0]/10:1]
Yv = Y[0:Y.shape[0]/10:1]

Xtrain = X[X.shape[0]/10::1]
Ytrain = Y[Y.shape[0]/10::1]

print '\033[94m'+"Validation:"+'\033[0m'

for i in [0.00001, 0.001, 1, 5, 10]:
    models = train_multiclass(Xtrain, Ytrain, '-g 0.05 -c '+str(i)+' -q')
    acc, pred, actual = test_multiclass(models, Xv, Yv)
    print "Validation Accuracy with C = ", i, " is : ", acc*100
    acc, pred, actual = test_multiclass(models, Xtest, Ytest)
    print "Test Accuracy with C = ", i, " is : ", acc*100