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

def generate_vocab():
    seen = set()
    values = []
    output = []
    for value in json_reader("train.json"):
        values.extend(value["text"].strip().split())
    for word in values:
        if word not in seen:
            output.append(value)
            seen.add(value)
    print output
    return output

vocab = generate_vocab()