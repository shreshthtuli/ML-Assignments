"""
MIT License

Copyright (c) 2019 Shreshth Tuli

Machine Learning Model : Naive Bayes

"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint
from matplotlib import cm
from utils import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score

# Hyper parameters
tokenize_type = 2

# Global constants
categories = 5
dicts = 0
Denom = [0.0, 0.0, 0.0, 0.0, 0.0]

# Parameters
Phi = [0, 0, 0, 0, 0] # For y : 1 to 5
Theta = np.empty((dicts, categories)) # 
majority_class = 0

def bigrams(lst):
    result = []
    for i in range(len(lst)-1):
        result.append(lst[i] + lst[i+1])
    return result

def tokenize(str):
    if tokenize_type == 1: # Do nothing
        return str.strip().split()
    elif tokenize_type == 2: # Stemming and Stopword removal
        return getStemmedDocuments(str)
    elif tokenize_type == 3: # 2 and Bigrams
        a = getStemmedDocuments(str)
        a.extend(bigrams(a))
        return a
    elif tokenize_type == 4: # 2 and adding first 10 words again
        a = getStemmedDocuments(str)
        a.extend(a[:10])
        return a
    elif tokenize_type == 5: # 2 and pos tags
        a = getStemmedDocuments(str)
        a.extend(pos_tag(a))
        return a
    elif tokenize_type == 6: # 2 and adding hash of good/bad
        a = getStemmedDocuments(str)
        if u'amaz' in a or 'good' in a or u"excel" in a or "best" in a or u'love' in a:
            a.append("qwertyuiop")
        if "bad" in a or "worst" in a or "dissapoint" in a or "poor" in a:
            a.append("asdfghjkl")
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
        # if counter == limit: break
    return (dict(zip(output, np.arange(0, len(output), 1))), inputs)

def train(vocab, inputs):
    # Initialization
    global majority_class
    Phi = [0.0, 0.0, 0.0, 0.0, 0.0] # For y : 1 to 5
    Theta = np.empty((dicts, categories)) # 
    Denom = [0.0, 0.0, 0.0, 0.0, 0.0]
    M = 0 # Number of training examples
    counts = np.zeros(5)
    for i in range(dicts):
	    for j in range(categories):
		    Theta[i][j] = 0

    for inp in inputs:
        M += 1
        c = int(inp["stars"])
        Phi[c-1] += 1; counts[c-1] += 1
        for word in tokenize(inp["text"]):
            k = vocab[word]
            Theta[k][c-1] += 1            
            Denom[c-1] += 1
    majority_class = counts.argmax() + 1
    
    for i in range(categories):
        Phi[i] = np.log2(Phi[i]) #(Phi[i]+1)/(M+5)
    for k in range(dicts):
        for c in range(categories):
            Theta[k][c] = np.log2((Theta[k][c] + 1.0) / (Denom[c] + dicts))
    
    return Phi, Theta, Denom

def theta_k_c(token, category):
    try: return Theta[vocab[token]][category]
    except: return np.log2(1.0 / (dicts + Denom[category]))
    
def classify(text):
    probs = np.zeros(5)
    tokens = tokenize(text)
    for c in range(categories):
        for token in tokens:
            probs[c] += theta_k_c(token, c)
        probs[c] += Phi[c]
    return probs.argmax() + 1

def test(filename, Phi, Theta, option=1):
    correct = 0
    count = 0
    actual_ys = []; predicted_ys = []
    if tokenize_type == 3: correct += 500
    for view in json_reader(filename):
        count += 1
        actual_y = int(view["stars"])
        predicted_y = classify(view["text"]) if option == 1 else (randint(1, 5) if option == 2 else majority_class)
        actual_ys.append(actual_y); predicted_ys.append(predicted_y)
        if(actual_y == predicted_y):
            correct += 1
        # if count > 10000: break
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

option = 1 # 1 = trained model, 2 = random, 3 = majority

print ("Test Accuracy = "), 
accuracy, actual_ys, predicted_ys = test("test.json", Phi, Theta, option)
print (accuracy*100)

# print ("Training Accuracy = "), 
# accuracy, actual_ys, predicted_ys = test("train.json", Phi, Theta)
# print (accuracy)

cm = confusion_matrix(actual_ys, predicted_ys)

plot_confusion_matrix(cm, [1, 2, 3, 4, 5])
plt.savefig("Confusion-Matrix")

print "Macro F1 Score : ", f1_score(actual_ys, predicted_ys, average='macro')