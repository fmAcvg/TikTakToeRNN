import numpy as np


def compute_optimal_accuracy(prediction, target_multi_hot):
    """
    Neue Accuracy-Metrik für Multi-Hot Targets:
    Zählt korrekt wenn prediction in den optimalen Moves liegt
    
    Args:
        prediction: Index des vorhergesagten Zugs (0-8)
        target_multi_hot: 9-dim vektor mit 1 für alle optimalen Moves
    
    Returns: 1 wenn korrekt, 0 sonst
    """
    return 1.0 if target_multi_hot[int(prediction)] == 1 else 0.0


def train_on_data_set(model, current_position, correct_move, learning_rate, weight_decay=0.0, legal_move_mask=None, multi_hot=False):
    """
    Trainiert auf einem Sample mit Softmax-CE auf linearen Output-Logits.
    
    Args:
        model: NeuralNetwork-Instanz
        current_position: Board-State (9 Felder, kanonisiert)
        correct_move: 
            - Wenn multi_hot=False: One-Hot-Vektor (9 Ausgänge) ODER Index
            - Wenn multi_hot=True: Multi-Hot-Vektor (9 Ausgänge, 1 für alle optimalen Moves)
        learning_rate: Lernrate
        weight_decay: L2-Regularisierung (Standard: 0)
        legal_move_mask: Optional - Maske der legalen Züge (9er-Vektor mit 1 für legal, 0 für illegal)
        multi_hot: Ob target Multi-Hot Format ist (mehrere optimale Moves)
    """
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

    # Softmax on raw logits for loss & gradients
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)

    # Correct prediction on masked logits
    logits_masked = logits.copy()
    if legal_move_mask is not None:
        logits_masked[legal_move_mask == 0] = -np.inf
    exps_masked = np.exp(logits_masked - np.max(logits_masked))
    probs_masked = exps_masked / np.sum(exps_masked)

    # Normalize correct_move wenn nötig
    if multi_hot:
        # Multi-Hot: Normalisiere auf Wahrscheinlichkeitsverteilung über optimale Moves
        target_probs = np.array(correct_move, dtype=float)
        if np.sum(target_probs) > 0:
            target_probs = target_probs / np.sum(target_probs)
        loss = -np.sum(target_probs * np.log(probs + 1e-15))
        # Accuracy: Ist prediction in den optimalen Moves?
        pred_move = np.argmax(probs_masked)
        correct = int(correct_move[pred_move] == 1)
    else:
        # Single-hot: Altes Format
        target_probs = np.array(correct_move, dtype=float)
        if np.sum(target_probs) > 0:
            target_probs = target_probs / np.sum(target_probs)
        loss = -np.sum(target_probs * np.log(probs + 1e-15))
        correct = (np.argmax(probs_masked) == np.argmax(target_probs))

    # Gradient auf Logits (für Multi-Hot oder Single-Hot)
    grad = probs - target_probs

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
