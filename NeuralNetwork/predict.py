from rnn.Layer import Layer
from rnn.neuron import Neuron
import numpy as np

save_weights = True


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inputs through each layer sequentially and return raw outputs.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, board: np.ndarray):
        """
        Return (predicted_move_index, probabilities) using softmax.
        """
        output = self.forward(board)
        # stable softmax
        exps = np.exp(output - np.max(output))
        probs = exps / np.sum(exps)
        predicted_move = int(np.argmax(probs))
        return predicted_move, probs

    def save_model(self, filepath: str = "./weights/") -> None:
        """
        Save weights and biases from each neuron into a .npz file.

        The filename mirrors the old script's datetime-based naming.
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
        filename_date_time = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        filename = f'model_{filename_date_time}.npz'
        try:
            np.savez(filename, **weights)
        except Exception as e:
            print(f"error saving model: {e}")


def create_layer(input_size: int, num_neurons: int) -> Layer:
    neurons = []
    for _ in range(num_neurons):
        weights = np.random.randn(input_size)
        bias = np.random.randn()
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

