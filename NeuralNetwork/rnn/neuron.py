import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias
        self.output = None
        self.inputs = None
        self.z = None  # pre-activation value
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.tanh(self.z)
        return self.output


