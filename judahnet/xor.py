"""
The canonical example of a function that can't be 
learned with a simple linear model is XOR
"""
import numpy as np

from judahnet.train import train
from judahnet.nn import NeuralNet
from judahnet.layers import Linear, Tanh

inputs = np.array([
     [0, 0],
     [1, 0],
     [0, 1],
     [1, 1]   
])

targets = np.array([
    [1, 0],
    [0, 1]
])