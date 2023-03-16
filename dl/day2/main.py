import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and inputs
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # remember the input values
        self.inputs = inputs
        # calculate output values from inputs , weights and biases
        self.output = np.dot(inputs, self.weights)


class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # applying ReLu activation function
        self.output = np.maximum(0, inputs)


class Activation_SoftMax:

    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # storing the output
        self.output = probabilities
