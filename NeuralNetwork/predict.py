from .rnn.Layer import Layer
from .rnn.neuron import Neuron
import numpy as np

save_weights = False


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass mit linearem Output-Layer (Logits).
        Hidden-Layer: tanh
        Output-Layer: linear
        """
        if len(self.layers) == 0:
            return np.array(inputs, dtype=float)

        x = np.array(inputs, dtype=float)

        # Hidden layers
        for layer in self.layers[:-1]:
            x = layer.forward(x)

        # Output layer (linear logits, KEIN tanh)
        output_layer = self.layers[-1]
        logits = np.array([np.dot(neuron.weights, x) + neuron.bias for neuron in output_layer.neurons])
        return logits

    def predict(self, board: np.ndarray, legal_mask=None):
        output = self.forward(board)
        # stable softmax
        exps = np.exp(output - np.max(output))
        probs = exps / np.sum(exps)
        
        if legal_mask is not None:
            # Set illegal moves to -inf before argmax
            masked_output = output.copy()
            masked_output[legal_mask == 0] = -np.inf
            predicted_move = int(np.argmax(masked_output))
        else:
            predicted_move = int(np.argmax(probs))
        
        return predicted_move, probs

    def save_model(self, filepath: str = "./models/") -> None:
        """
        Save weights and biases from each neuron into a .npz file.

        The filename mirrors the old script's datetime-based naming.
        """
        import os
        os.makedirs(filepath, exist_ok=True)
        weights = {}
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
        filename_date_time = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        filename = os.path.join(filepath, f'model_{filename_date_time}.npz')
        try:
            np.savez(filename, **weights)
            print(f"Model gespeichert: {filename}")
        except Exception as e:
            print(f"error saving model: {e}")


def create_layer(input_size: int, num_neurons: int) -> Layer:
    neurons = []
    # Xavier/Glorot initialization for tanh
    limit = np.sqrt(6 / (input_size + num_neurons))
    for _ in range(num_neurons):
        weights = np.random.uniform(-limit, limit, input_size)
        bias = np.zeros(1)[0]
        neurons.append(Neuron(weights, bias))
    return Layer(neurons)


if __name__ == '__main__':
    # Example board and a small randomly initialized network to demonstrate usage.
    board = np.array([
        1,  0, -1,
        0,  1,  0,
       -1,  0,  0
    ])

    # Layer sizes (same as prior script)
    input_size = 9
    hidden_size = 16
    output_size = 9

    layer1 = create_layer(input_size, hidden_size)
    layer2 = create_layer(hidden_size, hidden_size)
    layer3 = create_layer(hidden_size, hidden_size)
    output_layer = create_layer(hidden_size, output_size)

    layers = [layer1, layer2, layer3, output_layer]

    net = NeuralNetwork(layers)
    predicted_move, probs = net.predict(board)
    print(f"predicted move index: {predicted_move}")

    if save_weights:
        net.save_model()

