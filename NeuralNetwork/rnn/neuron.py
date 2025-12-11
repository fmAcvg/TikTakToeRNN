import numpy

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = None 
    
    def tanh(self, x):
        return numpy.tanh(x)
    
    def forward(self, inputs):
        self.output = self.tanh(numpy.dot(self.weights, inputs) + self.bias)
        return self.output
    




