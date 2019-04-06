"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Neural Networks

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from itertools import chain
from time import time
from sklearn.metrics import confusion_matrix
import csv

# np.set_printoptions(threshold=np.inf)

SIGMOID = True

def parseData(filename):
    file = open(filename, "r")
    return np.matrix([map(int, line.strip().split(',')) for line in file.readlines()])

def values(attribute):
    if attribute == 10: return 10 # Y
    if attribute%2 == 0: return 4 # Suit
    else: return 13 # Rank

def oneHot(rng, val):
    lst = [0] * rng;
    if rng == 10: lst[val] = 1;
    else: lst[val-1] = 1
    return lst

def storeMatrix(mat, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(mat)

def preProcessFile(readFile, storeFile):
    X = parseData(readFile)
    res = [list(chain.from_iterable(oneHot(values(j), X[i,j]) for j in range(X.shape[1]))) \
        for i in range(X.shape[0])]
    storeMatrix(res, storeFile)

def activation(x):
    if SIGMOID: return 1.0 / (1.0 + np.exp(-x))
    x[x <= 0] = 0
    return x # ReLU

def finalActivation(x):
    return 1.0 / (1.0 + np.exp(-x))

def sumDifference(arr, e):
    sum = 0
    for i in range(e, e-10, -1):
        try: sum += abs(arr[i] - arr[i-1])
        except: pass
    return sum

def plotCost(graph, arch="NN"):
    plt.xlabel('Number of Epochs')
    plt.title("MSE cost with epochs for "+arch)
    plt.ylabel("MSE Cost")
    plt.plot(graph)
    plt.savefig("Cost with time "+arch+'.png', bbox_inches='tight')
    plt.show()
    plt.clf()

def plotGraph(title, ylbl, x, y):
    plt.xlabel('Number of Nodes')
    plt.title(title)
    plt.ylabel(ylbl)
    plt.plot(x, y, '-ro')
    plt.savefig(ylbl+'.png', bbox_inches='tight')
    plt.show()
    plt.clf()

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

class Layer():
    def __init__(self,in_size,out_size):
        if(not in_size == -1):
            self.w = np.random.randn(in_size, out_size-1)
        self.delta = 0; self.out = 0

class NeuralNet():
    def __init__(self,in_size,out_size,layers):
        self.layers = []; layers = [in_size]+layers
        self.layers.append(Layer(-1,in_size+1))
        for i in range(1,len(layers)):
            self.layers.append(Layer(layers[i-1]+1,layers[i]+1))
        self.layers.append(Layer(layers[-1]+1, out_size+1))

    def sigmoidPrime(self, s):
        return np.multiply(s, 1-s)
    
    def activationDer(self, s):
        if SIGMOID: return np.multiply(s, 1-s)
        return np.array(s > 0)

    def forward(self,x):
        self.layers[0].out = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        for idx in range(1,len(self.layers)-1):
            self.layers[idx].out = activation(np.dot(self.layers[idx-1].out,self.layers[idx].w))
            self.layers[idx].out = np.concatenate((np.ones((self.layers[idx].out.shape[0],1)),self.layers[idx].out),axis=1)
        self.layers[-1].out = finalActivation(np.dot(self.layers[-2].out,self.layers[-1].w))
        return self.layers[-1].out
    
    def backprop(self, y, eta):
        for idx, layer in enumerate(self.layers[::-1]):
            if(idx == 0):
                layer.delta = -1 * np.multiply((y - layer.out), self.sigmoidPrime(layer.out))
                # print "Last layer:", y, "\n",  layer.out, "\n", -layer.delta
            else:
                next_layer = self.layers[-idx]
                layer.delta = np.multiply(next_layer.delta.dot(next_layer.w[1:,:].T), self.activationDer(layer.out[:,1:]))
            if not SIGMOID: layer.delta = np.clip(layer.delta, -10, 10)
        for idx in range(1,len(self.layers)):
            self.layers[idx].w -= eta * self.layers[idx-1].out.T.dot(self.layers[idx].delta)

    def predict(self, x):
        return self.forward(x).argmax()
    
    def fit(self, x_train, y_train, eta2, epochs, batchsize=1, adaptive=False):
        batches = int(x_train.shape[0]/batchsize); eta = eta2
        costArr = []; totalCost = 0; prevCost = 0; last = False
        for e in range(epochs):
            totalCost = 0
            for i in range(batches):
                self.forward(x_train[i*batchsize:(i+1)*batchsize])
                self.backprop(y_train[i*batchsize:(i+1)*batchsize], eta)
                totalCost +=  (np.linalg.norm(y_train[i*batchsize:(i+1)*batchsize] - self.layers[-1].out)**2).sum()/(2*batchsize)
            # print "Cost:", totalCost, "Epoch:", e; 
            if adaptive and totalCost > prevCost + 0.0001: 
                if last: eta = eta/5
                else: last = True
            if adaptive and totalCost <= prevCost + 0.0001:
                last = False
            prevCost = totalCost
            if sumDifference(costArr, e) < 0.1 and e > 10: print "Epochs =", e; break;
            costArr.append(totalCost)
        return costArr
    
    def accuracy(self, x_test, y_test, printCM=False):
        correct = 0; pred = []; actual = []
        for i in range(x_test.shape[0]):
            pred.append(self.predict(x_test[i]))
            actual.append(y_test[i].argmax())
            if pred[-1] == actual[-1]:
                correct += 1
        if printCM:
            cm = confusion_matrix(actual, pred); print cm
            plot_confusion_matrix(cm,[0,1,2,3,4,5,6,7,8,9])
        return 100*float(correct)/x_test.shape[0]

# Preprocess files

if argv[1] == '0':
    print '\033[95m'+"Parsing and preprocessing data"+'\033[0m'

    preProcessFile("poker-hand-training-true.data", "train.data")
    preProcessFile("poker-hand-testing.data", "test.data")


# Neural Network

print '\033[95m'+"Parsing data"+'\033[0m'

Data = parseData("train.data")
testData = parseData("test.data")

x_train = Data[:,:-10]
y_train = Data[:,-10:]

x_test = testData[:,:-10]
y_test = testData[:,-10:]

test = [5,10,15,20,25]
train_acc = []; test_acc = []; t = []

SIGMOID = False

print '\033[95m'+"Single hidden layer testing"+'\033[0m'

for units in test:
    print '\033[94m'+"Number of nodes = "+str(units)+'\033[0m'
    start = time()
    nn = NeuralNet(x_train.shape[1], 10, [units])
    graph = nn.fit(x_train, y_train, 0.1, 2000, 10, True)
    t.append(time()-start)
    train_acc.append(nn.accuracy(x_train, y_train))
    test_acc.append(nn.accuracy(x_test, y_test, True))
    plt.savefig("Confusion-Matrix "+"Single layer with nodes = "+str(units))
    plt.clf()
    print "Training accuracy = ", train_acc[-1]
    print "Test accuracy = ", test_acc[-1]
    print "Time = ", t[-1]
    plotCost(graph, "Single layer with nodes = "+str(units))

plotGraph("Training accuracy with nodes (Single hidden layer)", "Training accuracy %", test, train_acc)
plotGraph("Test accuracy with nodes (Single hidden layer)", "Test accuracy %", test, test_acc)
plotGraph("Time with nodes (Single hidden layer)", "Time in seconds", test, t)

print '\033[95m'+"Two hidden layers testing"+'\033[0m'

train_acc = []; test_acc = []; t = []

for units in test:
    print '\033[94m'+"Number of nodes = "+str(units)+'\033[0m'
    start = time()
    nn = NeuralNet(x_train.shape[1], 10, [units, units])
    graph = nn.fit(x_train, y_train, 0.1, 2000, 10, True)
    t.append(time()-start)
    train_acc.append(nn.accuracy(x_train, y_train))
    test_acc.append(nn.accuracy(x_test, y_test, True))
    plt.savefig("Confusion-Matrix "+"Two layer with nodes = "+str(units))
    plt.clf()
    print "Training accuracy = ", train_acc[-1]
    print "Test accuracy = ", test_acc[-1]
    print "Time = ", t[-1]
    plotCost(graph, "Two layer with nodes = "+str(units))

plotGraph("Training accuracy with nodes (Two hidden layers)", "Training accuracy %", test, train_acc)
plotGraph("Test accuracy with nodes (Two hidden layers)", "Test accuracy %", test, test_acc)
plotGraph("Time with nodes (Two hidden layers)", "Time in seconds", test, t)


