#!/usr/bin/env python
"""
Final Training with Augmented Dataset
- Uses 6224 augmented training samples
- Better hyperparameters
- Multi-Hot format
"""
import numpy as np
import os
from datetime import datetime
from NeuralNetwork.predict import NeuralNetwork, create_layer
from NeuralNetwork.training import train_on_data_set

print("="*70)
print("TicTacToe RNN Training - FINAL VERSION")
print("="*70)

CONFIG = {
    'hidden_layers': 3,
    'neurons_per_layer': 256,     # Large capacity (2x V2)
    'learning_rate': 0.0003,      # Even slower (0.0003)
    'weight_decay': 0.0005,       # Stronger L2 reg
    'epochs': 500,
    'early_stopping_patience': 150, # Even higher patience
    'batch_size': 32,
}

print(f"\nKonfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# ============================================================================
# 1. LOAD AUGMENTED DATASET
# ============================================================================
dataset_path = "datasets/tictactoe_dataset_augmented.npy"

if not os.path.exists(dataset_path):
    print(f"ERROR: {dataset_path} not found!")
    print("Run: python generate_augmented_dataset.py")
    exit(1)

dataset = np.load(dataset_path)
print(f"Datensatz geladen: {dataset.shape} ({len(dataset)} Samples)")

# Board-level split (unique boards only)
np.random.seed(42)
boards_only = dataset[:, :9]
unique_boards, unique_indices = np.unique(boards_only, axis=0, return_index=True)

print(f"Unique Boards: {len(unique_boards)}")

np.random.shuffle(unique_indices)
split_idx = int(len(unique_indices) * 0.8)
train_indices = unique_indices[:split_idx]
test_indices = unique_indices[split_idx:]

train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]

print(f"Train Set: {len(train_dataset)}")
print(f"Test Set: {len(test_dataset)}")
print()

# ============================================================================
# 2. CREATE MODEL
# ============================================================================
print("Erstelle Modell...")
layers = []
prev_neurons = 9

for i in range(CONFIG['hidden_layers']):
    layer = create_layer(prev_neurons, CONFIG['neurons_per_layer'])
    layers.append(layer)
    prev_neurons = CONFIG['neurons_per_layer']

output_layer = create_layer(prev_neurons, 9)
layers.append(output_layer)

model = NeuralNetwork(layers)
print(f"  Modell: 9 -> {CONFIG['hidden_layers']}x{CONFIG['neurons_per_layer']} -> 9")
print()

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================
print("Starte Training...")
print(f"  LR: {CONFIG['learning_rate']}")
print()

best_test_acc = 0
epochs_without_improvement = 0
best_model_path = None

for epoch in range(CONFIG['epochs']):
    # TRAINING
    train_loss = 0.0
    train_correct = 0
    
    for sample in train_dataset:
        board = sample[:9].astype(float)
        target = sample[9:].astype(int)
        
        loss, correct = train_on_data_set(
            model, board, target,
            learning_rate=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
            multi_hot=True
        )
        
        train_loss += loss
        if correct:
            train_correct += 1
    
    train_loss /= len(train_dataset)
    train_acc = 100.0 * train_correct / len(train_dataset)
    
    # TESTING
    test_loss = 0.0
    test_correct = 0
    
    for sample in test_dataset:
        board = sample[:9].astype(float)
        targets = sample[9:].astype(int)
        
        # Forward pass
        activation = np.array(board, dtype=float)
        for layer in model.layers[:-1]:
            activation = layer.forward(activation)
        
        # Output layer
        output_layer = model.layers[-1]
        logits = np.array([np.dot(n.weights, activation) + n.bias for n in output_layer.neurons])
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        # Loss
        target_probs = np.array(targets, dtype=float)
        if np.sum(target_probs) > 0:
            target_probs = target_probs / np.sum(target_probs)
        test_loss += -np.sum(target_probs * np.log(probs + 1e-15))
        
        # Accuracy
        pred_move = np.argmax(probs)
        if targets[pred_move] == 1:
            test_correct += 1
    
    test_loss /= len(test_dataset)
    test_acc = 100.0 * test_correct / len(test_dataset)
    
    # EARLY STOPPING
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        epochs_without_improvement = 0
        
        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        model_path = f"models/model_final_{timestamp}.npz"
        
        weights = {}
        for i, layer in enumerate(model.layers):
            for j, neuron in enumerate(layer.neurons):
                weights[f"layer_{i}_neuron_{j}_weights"] = neuron.weights
                weights[f"layer_{i}_neuron_{j}_bias"] = neuron.bias
        np.savez(model_path, **weights)
        best_model_path = model_path
        
        print(f"Epoch {epoch+1:4d} | Train: {train_acc:6.1f}% | Test: {test_acc:6.1f}% [BEST -> {model_path}]")
    else:
        epochs_without_improvement += 1
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d} | Train: {train_acc:6.1f}% | Test: {test_acc:6.1f}%")
    
    if epochs_without_improvement >= CONFIG['early_stopping_patience']:
        print(f"\n[OK] Early Stopping bei Epoch {epoch+1}")
        break

print()
print("="*70)
print(f"TRAINING COMPLETE")
print(f"Best Test Accuracy: {best_test_acc:.1f}%")
print(f"Best Model: {best_model_path}")
print("="*70)

# ============================================================================
# 4. SUMMARY
# ============================================================================
if best_test_acc >= 80:
    result = "EXCELLENT! Model learned optimal play!"
elif best_test_acc >= 60:
    result = "GOOD! Model has solid strategy."
elif best_test_acc >= 40:
    result = "FAIR! Model has some strategy."
else:
    result = "POOR! Model needs more work."

print(f"\nResult: {result}")
print(f"\nNext: Test in GUI using {os.path.basename(best_model_path)}")
