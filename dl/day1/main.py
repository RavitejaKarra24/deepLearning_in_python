import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self,n_inputs, n_neurons):
        # Initialize weights and inputs
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # remember the input values
        self.inputs = inputs
        # calculate output values from inputs , weights and biases
        self.output = np.dot(inputs,self.weights) + biases

X,y = spiral_data(samples=100, classes=3)
layer1 = Layer_Dense(3,4)
layer1.forward(X)
print(layer1.output[:5])
