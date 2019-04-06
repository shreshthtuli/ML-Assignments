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
    lst[val-1] = 1
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
    return max(0, x) # ReLU

def derivative(x):
    if SIGMOID: return activation(x) (1 - activation(x))
    return 1 if x >= 0 else 0 # ReLU


class Layer():
    def __init__(self,in_size,out_size):
        if(not in_size == -1):
            self.w = np.random.normal(0,0.1,in_size * (out_size-1)).reshape(in_size,out_size-1)
        self.delta = 0; self.out = 0

class NeuralNet():
    def __init__(self,in_size,out_size,layers):
        self.layers = []; layers = [in_size]+layers
        self.layers.append(Layer(-1,in_size+1))
        for i in range(1,len(layers)):
            self.layers.append(Layer(layers[i-1]+1,layers[i]+1)); print layers[i-1]+1, layers[i]
        self.layers.append(Layer(layers[-1]+1, out_size+1))

    def forward(self,x):
        self.layers[0].out = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        for idx in range(1,len(self.layers)-1):
            self.layers[idx].out = activation(np.dot(self.layers[idx-1].out,self.layers[idx].w))
            self.layers[idx].out = np.concatenate((np.ones((self.layers[idx].out.shape[0],1)),self.layers[idx].out),axis=1)
        self.layers[-1].out = activation(np.dot(self.layers[-2].out,self.layers[-1].w))
        return self.layers[-1].out
    
    def backprop(self, y, eta):
        for idx, layer in enumerate(self.layers[::-1]):
            if(idx == 0):
                layer.delta = -1 * np.array(y - layer.out) * np.array(layer.out) * np.array(1 - layer.out)
                # print "Last layer:", y, "\n",  layer.out, "\n", -layer.delta
            else:
                next_layer = self.layers[-idx]
                layer.delta = np.einsum('ij,kj->ki',next_layer.w[1:,:],next_layer.delta) * np.array(layer.out[:,1:]) * np.array(1 - layer.out[:,1:])
        for idx in range(1,len(self.layers)):
            self.layers[idx].w -= eta * np.einsum('ij,ik->jk',self.layers[idx-1].out,self.layers[idx].delta)

    def predict(self, x):
        return self.forward(x).argmax()
    
    def fit(self, x_train, y_train, eta, epochs, batchsize=1):
        batches = int(x_train.shape[0]/batchsize)
        costArr = []; totalCost = 0
        for e in range(epochs):
            totalCost = 0
            for i in range(batches):
                self.forward(x_train[i*batchsize:(i+1)*batchsize])
                self.backprop(y_train[i*batchsize:(i+1)*batchsize], eta)
                cost = (np.linalg.norm(y_train[i*batchsize:(i+1)*batchsize] - self.layers[-1].out)**2).sum()/(2*batchsize)
                totalCost += cost
                # print "Cost: ", cost; 
                # print x_train[i], self.forward(x_train[i]), self.predict(x_train[i]), y_train[i]
            print "Cost:", totalCost, "Epoch:", e; 
            if totalCost < 10: break;
            costArr.append(totalCost)
        return costArr
    
    def accuracy(self, x_test, y_test):
        correct = 0
        for i in range(x_test.shape[0]):
            pred = self.pred(x_test[i])
            actual = y_test[i]
            if pred == actual:
                count += 1
        return 100*float(correct)/x_test.shape[0]

# Preprocess files

if argv[1] == '0':
    print '\033[95m'+"Parsing and preprocessing data"+'\033[0m'

    preProcessFile("poker-hand-training-true.data", "train.data")
    preProcessFile("poker-hand-testing.data", "test.data")


# Neural Network

Data = parseData("train.data")
testData = parseData("test.data")

# Simple network
# x_train = np.matrix([[1,1],[0,1],[1,0],[0,0]])
# y_train = np.matrix([[1,0],[0,1],[0,1],[1,0]])

# nn = NeuralNet(x_train.shape[1], 2, [3,3])

# nn.fit(x_train, y_train, 0.5, 20000, 1)

# Real network
x_train = Data[:,:-10]
y_train = Data[:,-10:]

x_test = testData[:,:-10]
y_test = testData[:,-10:]

nn = NeuralNet(x_train.shape[1], 10, [20])
graph = nn.fit(x_train, y_train, 0.5, 100, 1)
plt.plot(graph)
plt.savefig("Cost with time"+'.png')
print nn.accuracy(x_train, y_train)
print nn.accuracy(x_test, y_test)


