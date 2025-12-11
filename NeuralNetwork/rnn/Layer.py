
from .neuron import Neuron


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
    
    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
        return outputs

