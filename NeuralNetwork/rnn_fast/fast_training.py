import numpy as np

def train_batch(model, X_batch, Y_batch, learning_rate, weight_decay=0.0, legal_masks_batch=None, multi_hot=False):
    """
    Trainiert Vektor-Batch auf einmal (1000x schneller).
    X_batch: Form (batch_size, 9)
    Y_batch: Form (batch_size, 9)
    legal_masks_batch: Form (batch_size, 9) oder None
    """
    batch_size = X_batch.shape[0]

    # --- FORWARD PASS ---
    activations = [X_batch]
    
    # Hidden layers
    for layer in model.layers[:-1]:
        z = np.dot(activations[-1], layer.weights) + layer.bias

        if getattr(layer, "activation", "tanh") == "relu":
            a = np.maximum(0.0, z)
        else:
            a = np.tanh(z)

        activations.append(a)
        
    # Output layer
    out_layer = model.layers[-1]
    logits = np.dot(activations[-1], out_layer.weights) + out_layer.bias
    
    # -- LOSS & ACCURACY ---
    # Softmax on batch
    # Trick for numerical stability over batch:
    logits_max = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - logits_max)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # Calculate Corrects (Accuracy masking handling)
    if legal_masks_batch is not None:
        logits_masked = logits.copy()
        logits_masked[legal_masks_batch == 0] = -1e9
        exps_m = np.exp(logits_masked - np.max(logits_masked, axis=1, keepdims=True))
        probs_masked = exps_m / np.sum(exps_m, axis=1, keepdims=True)
    else:
        probs_masked = probs

    if multi_hot:
        row_sums = np.sum(Y_batch, axis=1, keepdims=True)
        # Avoid division by zero if target is all 0 
        target_probs = np.where(row_sums > 0, Y_batch / row_sums, 0.0)
        
        # Loss per sample
        loss = -np.sum(target_probs * np.log(probs + 1e-15)) / batch_size
        
        preds = np.argmax(probs_masked, axis=1)
        # is prediction inside the multi_hot ones?
        # Trick: Wir nehmen np.arange um die richtigen Indices auszulesen
        correct = np.sum(Y_batch[np.arange(batch_size), preds] == 1)
        
    else:
        row_sums = np.sum(Y_batch, axis=1, keepdims=True)
        target_probs = np.where(row_sums > 0, Y_batch / row_sums, 0.0)
        loss = -np.sum(target_probs * np.log(probs + 1e-15)) / batch_size
        
        preds = np.argmax(probs_masked, axis=1)
        correct = np.sum(preds == np.argmax(Y_batch, axis=1))

    # --- BACKPROPAGATION ---
    # Ableitung der Loss w.r.t Logits ist (probs - target)
    d_z = (probs - target_probs) / batch_size # Skaliere gradient über Batch Size !!!
    
    # Backpropagate Output Layer
    a_prev = activations[-1] # Shape: (batch_size, last_hidden_neurons)
    
    # Gradients for W and b
    dW = np.dot(a_prev.T, d_z) + weight_decay * out_layer.weights # L2
    db = np.sum(d_z, axis=0)
    
    # Delta für den Layer davor
    d_aPrev = np.dot(d_z, out_layer.weights.T)
    
    # Backprop für OutputLayer abspeichern
    weight_updates = [(dW, db)]
    
    # Hidden Layers Rückwärts
    for i in range(len(model.layers)-2, -1, -1):
        layer = model.layers[i]
        
        a = activations[i+1]

        if getattr(layer, "activation", "tanh") == "relu":
            d_activation = (a > 0).astype(float)
        else:
            d_activation = 1.0 - a**2

        d_z = d_aPrev * d_activation
        a_prev = activations[i]
        
        dW = np.dot(a_prev.T, d_z) + weight_decay * layer.weights
        db = np.sum(d_z, axis=0)
        
        # Delta für DIESEN Input vorbereiten
        d_aPrev = np.dot(d_z, layer.weights.T)
        
        # Vorne anfügen (umgekehrte Reihenfolge)
        weight_updates.insert(0, (dW, db))

    # --- WEIGHT UPDATE ---
    for i, layer in enumerate(model.layers):
        dW, db = weight_updates[i]
        layer.weights -= learning_rate * dW
        layer.bias -= learning_rate * db
        
    return loss * batch_size, correct # gib zurück als summierte Loss für den Metric Counter
