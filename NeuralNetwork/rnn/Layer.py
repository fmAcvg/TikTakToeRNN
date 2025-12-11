import numpy as np
from .neuron import Neuron


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = np.array(inputs)
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return np.array(outputs)

