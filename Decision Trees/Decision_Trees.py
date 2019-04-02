"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Decision Trees

"""

import numpy as np
np.set_printoptions(threshold=np.inf)

MAX_NODES = 100000
NODES = 0

TreeNodes = []

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
    for i in range(5, 11):
        for j in range(X.shape[0]):
            X[j,i] = X[j,i] + 2

def plogp(num, den):
    return 0 if num == 0 else (num/den)*np.log2(num/den)

def getEntropy(D, attr, val):
    prob = 0.0
    ones = 0.0; zeros = 0.0
    if D.shape[1] == 0: return 0, 0
    for i in range(D.shape[0]):
        if D[i,attr] == val:
            prob += 1
            ones = ones + 1.0 if D[i,-1] == 1 else ones
            zeros = zeros + 1.0 if D[i,-1] == 0 else zeros
    prob = prob / D.shape[0]
    entropy = - plogp(ones, zeros+ones) \
              - plogp(zeros, zeros+ones)
    return prob, entropy

def values(attribute):
    values = []
    if attribute == 2: # Education
        values = range(7)
    elif attribute == 3: # marital status
        values = range(4)
    elif attribute in range(5, 11): # Repayment status
        values = range(12)
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
        global NODES
        self.parent = parent
        self.data = Dat
        self.attr = None
        self.children = []
        self.pruned = False
        self.entropy = 0
        self.zeros = 0
        self.ones = 0
        self.prediction = None
        NODES += 1
        self.isLeaf = True if NODES > MAX_NODES else False
        TreeNodes.append(self)
        self.findAttribute()
    
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
        self.entropy = - plogp(self.ones, self.data.shape[0]) \
                  - plogp(self.zeros, self.data.shape[0])

    def split(self):
        print "Splitting on attribute", self.attr, "into", len(values(self.attr)), "children"
        for value in values(self.attr):
            self.children.append(Node(self, extract(self.data, self.attr, value)))
    
    def findAttribute(self):
        self.predict()     
        informationArray = np.array([self.information(a) for a in range(23)])
        # print informationArray, informationArray.argmax()
        # exit(0)
        if np.max(informationArray) > 0.001 and NODES < MAX_NODES:
            self.attr = informationArray.argmax()
        else:
            self.isLeaf = True;
        if not self.isLeaf:
            self.split()
        
    def test(self, dat):
        if self.isLeaf:
            return self.prediction
        elif self.pruned:
            return self.parent.prediction
        else:
            return self.children[dat[self.attr]].test(dat)


def Test(tree, filename):
    TestData = parseData(filename)
    preProcessData(TestData)
    TestData = TestData.tolist()
    correct = 0; 
    for i in range(len(TestData)):
        pred = tree.test(TestData[i])
        actual = TestData[i][-1]
        correct = correct + 1 if pred == actual else correct
    print "Accuracy = ", 100*correct/(len(TestData)+0.0)
    print "Nodes = ", NODES
        

Data = parseData('credit-cards.train.csv')
preProcessData(Data)

Tree = Node(None, Data)
print Tree.prediction, Tree.zeros, Tree.ones, Tree.data.shape[0], Tree.attr

Test(Tree, "credit-cards.train.csv")
Test(Tree, "credit-cards.val.csv")
Test(Tree, "credit-cards.test.csv")