from rnn.network import Network
from rnn.neuron import Neuron
import numpy as np
import numpy as np

save_weights = True


def predict(network, inputs):
    return network.forward(inputs)

print("Predict module loaded.")

board = np.array([
    1,  0, -1,
    0,  1,  0,
   -1,  0,  0
])


def create_layer(input_size, num_neurons):
    neurons = []
    for _ in range(num_neurons):
        weights = np.random.randn(input_size)   # zufällige Weights
        bias = np.random.randn()                # zufälliger Bias
        neurons.append(Neuron(weights, bias))
    return Network(neurons)

# Layer-Größen
input_size = 9
hidden_size = 16
output_size = 9

layer1 = create_layer(input_size, hidden_size)
layer2 = create_layer(hidden_size, hidden_size)
layer3 = create_layer(hidden_size, hidden_size)
output_layer = create_layer(hidden_size, output_size)

layers = [layer1, layer2, layer3, output_layer]

def full_forward(layers, inputs):
    for layer in layers:
        inputs = layer.forward(inputs)
    return inputs

output = full_forward(layers, board)
probs =  np.exp(output) / np.sum(np.exp(output)) # Softmax
predicted_move = np.argmax(probs)

def save_model(layers, filepath="./weights/"):
    weights = {}
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
            weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
    filename_date_time = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
    filename = f'model_{filename_date_time}.npz'
    try: 
        np.savez(filename, **weights)
    except Exception as e:
        print(f"Error saving model: {e}")

if save_weights:
    save_model(layers)
print(f"Predicted move index: {predicted_move}")

