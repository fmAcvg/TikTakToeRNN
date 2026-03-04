import numpy as np


def train_on_data_set(model, current_position, correct_move, learning_rate, weight_decay=0.0):
    """Trainiert auf einem Sample mit Softmax-CE auf linearen Output-Logits."""
    # Forward bis zum letzten Hidden-Layer (Aktivierungen speichern)
    activations = [np.array(current_position, dtype=float)]

    if len(model.layers) == 0:
        return 0.0, False

    for layer in model.layers[:-1]:
        activations.append(layer.forward(activations[-1]))

    # Output-Layer als linear (logits), nicht tanh
    output_layer = model.layers[-1]
    output_input = activations[-1]
    logits = np.array([np.dot(neuron.weights, output_input) + neuron.bias for neuron in output_layer.neurons])

    # Softmax + CE
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    loss = -np.sum(correct_move * np.log(probs + 1e-15))
    correct = (np.argmax(probs) == np.argmax(correct_move))

    # Gradient auf Logits
    grad = probs - correct_move

    # Backprop Output-Layer (linear => kein tanh_grad)
    new_grad = np.zeros_like(output_input)
    for j, neuron in enumerate(output_layer.neurons):
        w_old = neuron.weights.copy()
        delta = grad[j]
        neuron.weights -= learning_rate * (delta * output_input + weight_decay * neuron.weights)
        neuron.bias -= learning_rate * delta
        new_grad += delta * w_old
    grad = new_grad

    # Backprop Hidden-Layer (tanh)
    for i in range(len(model.layers) - 2, -1, -1):
        layer = model.layers[i]
        layer_input = activations[i]
        new_grad = np.zeros_like(layer_input)

        for j, neuron in enumerate(layer.neurons):
            tanh_grad = neuron.tanh_derivative(neuron.z)
            delta = grad[j] * tanh_grad
            w_old = neuron.weights.copy()
            neuron.weights -= learning_rate * (delta * layer_input + weight_decay * neuron.weights)
            neuron.bias -= learning_rate * delta
            new_grad += delta * w_old

        grad = new_grad

    return loss, correct
