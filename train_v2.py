#!/usr/bin/env python
"""
Improved Training V2 - Better Hyperparameters and More Data
"""
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NeuralNetwork/training-dataset'))
from dataset import generate_optimal_dataset
from NeuralNetwork.predict import NeuralNetwork, create_layer
from NeuralNetwork.training import train_on_data_set

CONFIG = {
    'target_samples': 10000,      # Generate more data!
    'hidden_layers': 3,           # 3 instead of 2 (more capacity)
    'neurons_per_layer': 128,     # 128 instead of 64 (more capacity)
    'learning_rate': 0.0005,      # Even slower (0.0005 instead of 0.001)
    'weight_decay': 0.0001,
    'epochs': 500,                # Train much longer
    'early_stopping_patience': 100, # Higher patience
    'test_split': 0.2,
    'multi_hot': True,
}

print("="*70)
print("TicTacToe RNN Training - IMPROVED V2")
print("="*70)
print(f"\nKonfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# ============================================================================
# 1. DATASET
# ============================================================================
print("Generiere Datensatz...")
dataset = generate_optimal_dataset(CONFIG['target_samples'], balance_phases=True, multi_hot=True)
print(f"  Datensatz generiert: {dataset.shape}")

# Board-Level Split
np.random.seed(42)
unique_indices = np.arange(len(dataset))
np.random.shuffle(unique_indices)
split_idx = int(len(unique_indices) * CONFIG['test_split'])
test_indices = unique_indices[:split_idx]
train_indices = unique_indices[split_idx:]

test_dataset = dataset[test_indices]
train_dataset = dataset[train_indices]

print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print()

# ============================================================================
# 2. MODEL
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
print(f"  Modell: {[9] + [CONFIG['neurons_per_layer']]*CONFIG['hidden_layers'] + [9]}")
print()

# ============================================================================
# 3. TRAINING
# ============================================================================
print("Starte Training...")
print(f"  LR: {CONFIG['learning_rate']}")
print(f"  Epochs: {CONFIG['epochs']}")
print()

best_test_acc = 0
epochs_without_improvement = 0

for epoch in range(CONFIG['epochs']):
    # Training
    train_loss = 0
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
    train_acc = train_correct / len(train_dataset) * 100
    
    # Testing
    test_loss = 0
    test_correct = 0
    
    for sample in test_dataset:
        board = sample[:9].astype(float)
        targets = sample[9:].astype(int)
        
        activation = np.array(board, dtype=float)
        for layer in model.layers[:-1]:
            activation = layer.forward(activation)
        
        output_layer = model.layers[-1]
        logits = np.array([np.dot(neuron.weights, activation) + neuron.bias for neuron in output_layer.neurons])
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        target_probs = np.array(targets, dtype=float)
        if np.sum(target_probs) > 0:
            target_probs = target_probs / np.sum(target_probs)
        test_loss += -np.sum(target_probs * np.log(probs + 1e-15))
        
        pred_move = np.argmax(probs)
        if targets[pred_move] == 1:
            test_correct += 1
    
    test_loss /= len(test_dataset)
    test_acc = test_correct / len(test_dataset) * 100
    
    # Early Stopping
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        epochs_without_improvement = 0
        
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        model_path = f"models/model_v2_{timestamp}.npz"
        
        data = {}
        for i, layer in enumerate(model.layers):
            for j, neuron in enumerate(layer.neurons):
                data[f"layer_{i}_neuron_{j}_weights"] = neuron.weights
                data[f"layer_{i}_neuron_{j}_bias"] = neuron.bias
        np.savez(model_path, **data)
        
        print(f"Epoch {epoch+1:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}% [BEST]", flush=True)
    else:
        epochs_without_improvement += 1
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%")
    
    if epochs_without_improvement >= CONFIG['early_stopping_patience']:
        print(f"\n[OK] Early Stopping bei Epoch {epoch+1}")
        break

print()
print("="*70)
print(f"Final Best Test Accuracy: {best_test_acc:.1f}%")
print(f"Training Complete!")
print("="*70)
