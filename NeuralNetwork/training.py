import numpy as np


def train_on_data_set(model, current_position, correct_move, learning_rate):

    # Forward pass (speichere alle Aktivierungen)
    activations = [current_position]
    for layer in model.layers:
        activations.append(layer.forward(activations[-1]))
    
    # Softmax auf Output anwenden
    output = activations[-1]
    exps = np.exp(output - np.max(output))
    probs = exps / np.sum(exps)
    
    # Loss berechnen (Cross-Entropy)
    loss = -np.sum(correct_move * np.log(probs + 1e-15))
    
    # Backpropagation
    # Gradient nach Softmax (Cross-Entropy + Softmax)
    grad = probs - correct_move
    
    # Durch alle Layer rückwärts
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        layer_input = activations[i]
        new_grad = np.zeros_like(layer_input)
        
        for j, neuron in enumerate(layer.neurons):
            # Backprop durch tanh
            tanh_grad = neuron.tanh_derivative(neuron.z)
            delta = grad[j] * tanh_grad
            
            # Gewichte und Bias updaten
            neuron.weights -= learning_rate * delta * layer_input
            neuron.bias -= learning_rate * delta
            
            # Gradient für vorherige Layer
            new_grad += delta * neuron.weights
        
        grad = new_grad
    
    return loss

            