"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Naive Bayes

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
from random import shuffle, random
from utils import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
import nltk

# Hyper parameters
tokenize_type = 2

# Global constants
categories = 5
dicts = 0
Denom = [0.0, 0.0, 0.0, 0.0, 0.0]

# Parameters
Phi = [0, 0, 0, 0, 0] # For y : 1 to 5
Theta = np.empty((dicts, categories)) # 

def tokenize(str):
    if tokenize_type == 1:
        return str.strip().split()
    elif tokenize_type == 2:
        return getStemmedDocuments(str)
    elif tokenize_type == 3:
        a = getStemmedDocuments(str)
        a.extend(list(nltk.bigrams(str.split())))
        return a
    elif tokenize_type == 4:
        a = getStemmedDocuments(str)
        a.extend(a[:10])
        return a

def generate_vocab():
    seen = set()
    output = []
    inputs = []
    counter = 0; limit = 10000
    for value in json_reader("train.json"):
        inputs.append(value)
        for word in tokenize(value["text"]):
            if word not in seen:
                output.append(word)
                seen.add(word)
        counter+= 1
        if counter == limit: break
    return (dict(zip(output, np.arange(0, len(output), 1))), inputs)

def train(vocab, inputs):
    # Initialization
    Phi = [0.0, 0.0, 0.0, 0.0, 0.0] # For y : 1 to 5
    Theta = np.empty((dicts, categories)) # 
    Denom = [0.0, 0.0, 0.0, 0.0, 0.0]
    M = 0 # Number of training examples
    for i in range(dicts):
	    for j in range(categories):
		    Theta[i][j] = 0

    for inp in inputs:
        M += 1
        c = int(inp["stars"])
        Phi[c-1] += 1
        for word in tokenize(inp["text"]):
            k = vocab[word]
            Theta[k][c-1] += 1            
            Denom[c-1] += 1
    
    for i in range(categories):
        Phi[i] = (Phi[i]+1)/(M+5)
    for k in range(dicts):
        for c in range(categories):
            Theta[k][c] = (Theta[k][c] + 1) / (Denom[c] + dicts)
    
    return Phi, Theta, Denom

def theta_k_c(token, category):
    try: return Theta[vocab[token]][category]
    except: return 1.0 / (dicts + Denom[category])
    
def classify(text):
    probs = np.array([0, 0, 0, 0, 0])
    tokens = tokenize(text)
    for c in range(categories):
        for token in tokens:
            probs[c] += math.log(theta_k_c(token, c))
        probs[c] += math.log(Phi[c])
    return probs.argmax() + 1

def test(filename, Phi, Theta):
    correct = 0
    count = 0
    actual_ys = []; predicted_ys = []
    for view in json_reader(filename):
        count += 1
        actual_y = int(view["stars"])
        precited_y = classify(view["text"])
        actual_ys.append(actual_y); predicted_ys.append(precited_y)
        if(actual_y == precited_y):
            correct += 1
        if count > 10000: break
    return float(correct)/float(count), actual_ys, predicted_ys

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


vocab, inputs = generate_vocab()

dicts = len(vocab)

Phi, Theta, Denom = train(vocab, inputs)

print(Phi)
print(Theta)
print(Denom)

print ("Accuracy = "), 
accuracy, actual_ys, predicted_ys = test("test.json", Phi, Theta)
print (accuracy)

cm = confusion_matrix(actual_ys, predicted_ys)

print(cm)

# plot_confusion_matrix(cm, [1, 2, 3, 4, 5])
# plt.show()

print(classification_report(actual_ys, predicted_ys))