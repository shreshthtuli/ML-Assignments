"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Decision Trees

"""

import numpy as np
from sys import argv
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
np.set_printoptions(threshold=np.inf)

MAX_NODES = 100000
NODES = 0

CALC_LOCAL_MEDIAN = False

TreeNodes = []

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

def oneHot(rng, val):
    if rng == 2:
        return val
    lst = [0] * rng;
    lst[val] = 1
    return lst

def preProcessDataOneHot(X):
    preProcessData(X)
    res = []
    for i in range(X.shape[0]):
        a = []
        for j in range(X.shape[1]):
            a.append(oneHot(len(values(X[i,j])), X[i,j]))
        res.append(a)
    X = np.matrix(res)

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
        if CALC_LOCAL_MEDIAN and self.data.shape[1] != 0:
            preProcessData(self.data)
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
        for value in values(self.attr):
            self.children.append(Node(self, extract(self.data, self.attr, value)))
    
    def findAttribute(self):
        self.predict()     
        informationArray = np.array([self.information(a) for a in range(23)])
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
    return 100*correct/(len(TestData)+0.0)

def Validate(tree, TestData):
    correct = 0; 
    for i in range(len(TestData)):
        pred = tree.test(TestData[i])
        actual = TestData[i][-1]
        correct = correct + 1 if pred == actual else correct
    return 100*correct/(len(TestData)+0.0)

def giveXY(preProcessFunction, filename):
    Data = parseData(filename)
    preProcessFunction(Data)
    return Data[:,:-1], Data[:,-1]

def sumEqual(a, b):
    sum = 0.0
    for i in range(len(a)):
        if a[i] == b[i,0]: sum += 1
    return sum

def printBest(best, dtrees, train_acc, valid_acc, test_acc):
    print "Decision Tree Model =", dtrees[best] 
    print "Train Accuracy =", train_acc[best]
    print "Validation Accuracy =", valid_acc[best]
    print "Test Accuracy =", test_acc[best]
        
## Parsing and preprocessing

print '\033[95m'+"Parsing and preprocessing training data"+'\033[0m'

Data = parseData('credit-cards.train.csv')
preProcessData(Data)

## Training

if argv[1] == '1':

    print '\033[95m'+"Training Decision tree"+'\033[0m'

    Tree = Node(None, Data)
    print Tree.prediction, Tree.zeros, Tree.ones, Tree.data.shape[0], Tree.attr

    print "Nodes = ", NODES
    print "Train accuracy =", Test(Tree, "credit-cards.train.csv")
    print "Validation accuracy =", Test(Tree, "credit-cards.val.csv")
    print "Test accuracy =", Test(Tree, "credit-cards.test.csv")


## Pruning nodes based on validation

if argv[1] == '2':

    print '\033[95m'+"Pruning nodes based on validation accuracy"+'\033[0m'

    val = Test(Tree, "credit-cards.val.csv")
    newval = val

    PrunedArray = np.zeros(len(TreeNodes))
    TempArray = np.zeros(len(TreeNodes))

    TestData = parseData("credit-cards.val.csv")
    preProcessData(TestData)
    TestData = TestData.tolist()

    while True:
        TempArray = np.zeros(len(TreeNodes))
        for i in range(1, len(TreeNodes)):
            if PrunedArray[i] == 1:
                continue
            TreeNodes[i].pruned = True
            TempArray[i] = Validate(Tree, TestData)
            TreeNodes[i].pruned = False
        bestPrune = TempArray.argmax()
        TreeNodes[bestPrune].pruned = True
        val = newval
        newval = Validate(Tree, TestData)
        print "New validation accuracy =", newval, "by pruning node", bestPrune
        if newval <= val:
            TreeNodes[bestPrune].pruned = False; break
        PrunedArray[bestPrune] = 1

    print "Number of Nodes pruned =", np.count_nonzero(PrunedArray)
    print "Train accuracy =", Test(Tree, "credit-cards.train.csv")
    print "Validation accuracy =", Test(Tree, "credit-cards.val.csv")
    print "Test accuracy =", Test(Tree, "credit-cards.test.csv")


## Local median calculation

if argv[1] == '3':

    Data = parseData('credit-cards.train.csv')
    CALC_LOCAL_MEDIAN = True

    print '\033[95m'+"Local Data parsing tree"+'\033[0m'

    Tree = Node(None, Data)
    print Tree.prediction, Tree.zeros, Tree.ones, Tree.data.shape[0], Tree.attr

    print "Nodes = ", NODES
    print "Train accuracy =", Test(Tree, "credit-cards.train.csv")
    print "Validation accuracy =", Test(Tree, "credit-cards.val.csv")
    print "Test accuracy =", Test(Tree, "credit-cards.test.csv")

## Scikit-learn Decision Tree classifier

if argv[1] == '4' or argv[1] == '5':

    print '\033[95m'+"Scikit-learn Decision Tree classifier"+'\033[0m'

    criterion = ['entropy'] if argv[2] == 'simple' else ['gini', 'entropy']
    splitter = ['best'] if argv[2] == 'simple' else ['best', 'random']
    max_depth = [1, 5, 7, 10] + [None]
    min_samples_split = [0.001,0.01,0.1,5,10]
    min_samples_leaf = [0.001,0.01,0.1] + [1,5,10]
    max_features = [None] if argv[2] == 'simple' else [5,10,'sqrt','log2',None]
    min_impurity_decrease = [0] if argv[2] == 'simple' else [0.001, 0.01, 0.1]
    random_state = 0

    params = list(itertools.product(criterion,splitter,max_depth,min_samples_split,min_samples_leaf,max_features,min_impurity_decrease))

    dtrees = []
    for p in params:
        dtrees.append(DecisionTreeClassifier(criterion=p[0],splitter=p[1],max_depth=p[2],min_samples_split=p[3],min_samples_leaf=p[4],max_features=p[5],min_impurity_decrease=p[6],random_state=random_state))

    x_train, y_train = giveXY(preProcessData, 'credit-cards.train.csv')
    x_valid, y_valid = giveXY(preProcessData, 'credit-cards.val.csv')
    x_test, y_test = giveXY(preProcessData, 'credit-cards.test.csv')

    train_acc = []
    valid_acc = []
    test_acc = []

    print "Training", len(dtrees), "Decision Trees"; i = 0

    for tree in dtrees:
        tree.fit(x_train,y_train)
        if float(i)/len(dtrees)*100 % 10 == 0:
            print "Trained", i
        i += 1
        train_acc.append(100*sumEqual(tree.predict(x_train), y_train)/y_train.shape[0])
        valid_acc.append(100*sumEqual(tree.predict(x_valid), y_valid)/y_valid.shape[0])
        test_acc.append(100*sumEqual(tree.predict(x_test), y_test)/y_test.shape[0])
        

    params = np.array(params)
    dtrees = np.array(dtrees)
    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)
    test_acc = np.array(test_acc)

    if not argv[2] == 'simple':
        best = valid_acc[params[:,0] == 'gini'].argmax()
        printBest(best, dtrees, train_acc, valid_acc, test_acc)

    best = valid_acc[params[:,0] == 'entropy'].argmax() + int(dtrees.shape[0]/2)
    printBest(best, dtrees, train_acc, valid_acc, test_acc)

    if argv[1] == '5':
        print '\033[95m'+"Scikit-learn One-Hot Decision Tree classifier"+'\033[0m'
        x_train, y_train = giveXY(preProcessDataOneHot, 'credit-cards.train.csv')
        x_valid, y_valid = giveXY(preProcessDataOneHot, 'credit-cards.val.csv')
        x_test, y_test = giveXY(preProcessDataOneHot, 'credit-cards.test.csv')
        bestTree = dtrees[best]
        bestTree.fit(x_train, y_train)
        print "Training accuracy =", 100*sumEqual(tree.predict(x_train), y_train)/y_train.shape[0]
        print "Validation accuracy =", 100*sumEqual(tree.predict(x_valid), y_valid)/y_valid.shape[0]
        print "Test accuracy =", 100*sumEqual(tree.predict(x_test), y_test)/y_test.shape[0]