import numpy as np
from .fast_layer import FastLayer

class FastNeuralNetwork:
    """Ein High-Speed Vektor-Modell. Kompatibel mit der alten API (gleiche Gewichts-Ladefunktion), aber nutzt Matrizen"""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        """Batch forward pass"""
        x = np.atleast_2d(inputs)
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        # linear output for last layer
        x = self.layers[-1].forward_linear(x)
        return x

    def predict(self, board, legal_mask=None):
        """Macht Predictions für 1 Board (oder Batch). Für Gameplay."""
        output = self.forward(board)
        
        # stable softmax on last output only (if given a batch, assume batch size 1 for predictability compatibility here)
        out_single = output[0] if output.ndim == 2 else output
        
        exps = np.exp(out_single - np.max(out_single))
        probs = exps / np.sum(exps)

        if legal_mask is not None:
            # Set illegal moves to -inf before argmax
            masked_output = out_single.copy()
            masked_output[legal_mask == 0] = -np.inf
            predicted_move = int(np.argmax(masked_output))
        else:
            predicted_move = int(np.argmax(probs))

        return predicted_move, probs
        

def create_fast_layer(input_size: int, num_neurons: int, activation: str = "relu") -> FastLayer:
    return FastLayer(input_size, num_neurons, activation=activation)
