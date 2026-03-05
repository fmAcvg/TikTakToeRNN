import numpy as np

class FastLayer:
    """Vektorisierte Version eines Dense Layers (vollvernetzt)"""
    def __init__(self, input_size, num_neurons, activation="relu"):
        self.activation = activation

        # Initialization abhängig von Aktivierung
        if activation == "relu":
            # He init für ReLU
            scale = np.sqrt(2.0 / input_size)
            self.weights = np.random.randn(input_size, num_neurons) * scale
        else:
            # Xavier/Glorot für tanh/linear
            limit = np.sqrt(6 / (input_size + num_neurons))
            self.weights = np.random.uniform(-limit, limit, (input_size, num_neurons))

        # Matrix der Dimension (input_size, num_neurons)
        self.bias = np.zeros(num_neurons)
        
        self.inputs = None
        self.z = None
        self.output = None

    def forward(self, inputs):
        """Erwartet inputs als Liste von Arrays (Batch) oder Array: (batch_size, input_size)"""
        self.inputs = np.atleast_2d(inputs)
        # Vectorized Dot Product
        self.z = np.dot(self.inputs, self.weights) + self.bias

        if self.activation == "relu":
            self.output = np.maximum(0.0, self.z)
        elif self.activation == "tanh":
            self.output = np.tanh(self.z)
        else:
            self.output = self.z

        return self.output

    def forward_linear(self, inputs):
        """Für den letzten Layer (Logits) wo KEIN Tanh angewendet wird"""
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.z
        return self.output
