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

# Hyper parameters
tokenize_type = 1

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

def generate_vocab():
    seen = set()
    output = []
    inputs = []
    counter = 0; limit = 100000
    for value in json_reader("train.json"):
        inputs.append(value)
        for word in tokenize(value["text"]):
            if word not in seen:
                output.append(word)
                seen.add(word)
        counter+= 1
        # if counter == limit: break
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
    for view in json_reader(filename):
        count += 1
        actual_y = int(view["stars"])
        precited_y = classify(view["text"])
        if(actual_y == precited_y):
            correct += 1
        # if count > 10000: break
    return float(correct)/float(count)


vocab, inputs = generate_vocab()

dicts = len(vocab)

Phi, Theta, Denom = train(vocab, inputs)

print(Phi)
print(Theta)
print(Denom)

print ("Accuracy = "), 
print (test("test.json", Phi, Theta))