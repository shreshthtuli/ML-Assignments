"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Decision Trees

"""

import numpy as np
np.set_printoptions(threshold=np.inf)

def parseData(filename):
    file = open(filename, "r")
    lines = file.readlines()
    read = np.matrix([map(int, line.strip().split(',')) for line in lines[2:]])
    X = read[:,1:]
    return X

def preProcessData(X):
    for i in [0, 4] + range(11, 23):
        median = np.median(X[:,i], axis=0)
        for j in range(X.shape[0]):
            X[j,i] = 1 if X[j,i] >= median else 0 # Continuous to binary
    for j in range(X.shape[0]):
        X[j,1] = X[j,1] - 1 # Binary 1,2 to 0,1

def log2(val):
    return 0 if val == 0 else np.log2(val)

def getEntropy(D, attr, val):
    prob = 0.0
    ones = 0.0; zeros = 0.0
    for i in range(D.shape[0]):
        if D[i,attr] == val:
            prob += 1
            ones = ones + 1.0 if D[i,-1] == 1 else ones
            zeros = zeros + 1.0 if D[i,-1] == 0 else zeros
    prob = prob / D.shape[0]
    entropy = - (ones/D.shape[0])*log2(ones/D.shape[0]) \
              - (zeros/D.shape[0])*log2(zeros/D.shape[0])
    return prob, entropy

def values(attribute):
    values = []
    if attribute == 2: # Education
        values = range(7)
    elif attribute == 3: # marital status
        values = range(4)
    elif attribute in range(5, 11): # Repayment status
        values = range(-2, 10)
    else: # Binary or conitunuous
        values = range(2)
    return values

def extract(D, attr, val):
    dret = []
    D1 = D.tolist()
    for i in range(D.shape[0]):
        if D[i,attr] == val:
            dret.append(D1[i])
    return np.matrix(dret) 

class Node:
    def __init__(self, parent, Dat):
        self.parent = parent
        self.data = Dat
        self.attr = None
        self.children = []
        self.zeros = 0
        self.ones = 0
        self.entropy = 0
        self.prediction = None
        self.isLeaf = False
    
    def information(self, attribute):
        childEntropy = 0
        for value in values(attribute):
            prob, entr = getEntropy(self.data, attribute, value)
            childEntropy += prob * entr

        return self.entropy - childEntropy

    def predict(self):
        try: self.ones = np.count_nonzero(self.data[:,-1]) + 0.0 
        except: pass
        self.zeros = self.data.shape[0] - self.ones if self.data.shape[1] > 0 else 0
        self.prediction = 1 if self.ones > self.zeros else 0

    def split(self):
        for value in values(self.attr):
            self.children.append(Node(self, extract(self.data, self.attr, value)))
    
    def findAttribute(self):
        self.predict()
        self.entropy = - (self.ones/self.data.shape[0])*log2(self.ones/self.data.shape[0]) \
                  - (self.zeros/self.data.shape[0])*log2(self.zeros/self.data.shape[0])
        print self.entropy, np.array([self.information(a) for a in range(23)])
        self.attr = np.array([self.information(a) for a in range(23)]).argmax()
        if not self.isLeaf:
            self.split()
        

Data = parseData('credit-cards.train.csv')
preProcessData(Data)

Tree = Node(None, Data)
Tree.predict()
print Tree.prediction, Tree.zeros, Tree.ones, Tree.data.shape[0]
Tree.isLeaf = False
Tree.findAttribute()
print Tree.attr

for i in Tree.children:
    i.predict()
    print i.zeros, i.ones