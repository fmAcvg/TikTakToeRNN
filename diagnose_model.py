#!/usr/bin/env python
"""
Diagnose the trained model performance on the test set.
"""
import numpy as np
import os
import sys
sys.path.insert(0, 'NeuralNetwork/training-dataset')
from NeuralNetwork.predict import NeuralNetwork

# Load dataset
data = np.load("datasets/tictactoe_dataset_5000_optimal.npy")

# Split (same as training)
np.random.seed(42)
unique_indices = np.arange(len(data))
np.random.shuffle(unique_indices)
split_idx = int(len(unique_indices) * 0.8)
train_indices = unique_indices[:split_idx]
test_indices = unique_indices[split_idx:]

train_dataset = data[train_indices]
test_dataset = data[test_indices]

# Load the trained model
model_path = "models/model_improved_2026-03-04T22-25-54.npz"

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    models = sorted([m for m in os.listdir("models/") if m.startswith("model_improved")], reverse=True)
    if models:
        model_path = f"models/{models[0]}"
        print(f"Using latest model: {model_path}")
    else:
        print("No improved models found!")
        exit(1)

# Load weights properly
model_data = np.load(model_path)
layer_idx = 0
layers = []
from NeuralNetwork.rnn.neuron import Neuron
from NeuralNetwork.rnn.Layer import Layer

while True:
    keys = [k for k in model_data.keys() if k.startswith(f"layer_{layer_idx}_")]
    if not keys:
        break
    
    neurons = []
    neuron_count = len(set([k.split('_')[3] for k in keys]))
    for neuron_id in range(neuron_count):
        weight = model_data[f"layer_{layer_idx}_neuron_{neuron_id}_weights"]
        bias = model_data[f"layer_{layer_idx}_neuron_{neuron_id}_bias"]
        neurons.append(Neuron(weight, bias))
    layers.append(Layer(neurons))
    layer_idx += 1

model = NeuralNetwork(layers)

print("="*70)
print("MODEL DIAGNOSTIC")
print("="*70)
print(f"Model: {model_path}")
print(f"Layers: {len(layers)}")
print(f"Test Set Size: {len(test_dataset)}")
print()

# Test the model
correct_optimal = 0
correct_single = 0
pred_in_optimal = 0

for sample in test_dataset:
    board = sample[:9].astype(float)
    targets = sample[9:].astype(int)
    
    # Forward pass
    activation = np.array(board, dtype=float)
    for layer in model.layers[:-1]:
        activation = layer.forward(activation)
    
    # Output
    output_layer = model.layers[-1]
    logits = np.array([np.dot(neuron.weights, activation) + neuron.bias for neuron in output_layer.neurons])
    
    # Legal move masking
    legal_mask = (board == 0)
    logits_masked = logits.copy()
    logits_masked[~legal_mask] = -np.inf
    
    exps = np.exp(logits_masked - np.max(logits_masked))
    probs = exps / np.sum(exps)

    # Predictions
    pred_move = np.argmax(probs)
    pred_in_opt = (targets[pred_move] == 1)
    
    # Metrics
    if pred_in_opt:
        correct_optimal += 1
    pred_in_optimal += 1
    
    if np.sum(targets) == 1 and targets[pred_move] == 1:
        correct_single += 1

test_acc_optimal = correct_optimal / len(test_dataset) * 100
test_acc_single = correct_single / len(test_dataset) * 100

print("TEST RESULTS")
print("="*70)
print(f"Fair Accuracy (pred in optimal_moves): {test_acc_optimal:.1f}%")
print(f"Single-Move Positions Accuracy: {test_acc_single:.1f}%")
print(f"Tested Samples: {len(test_dataset)}")
print()
print("INTERPRETATION:")
if test_acc_optimal >= 80:
    print("✓ EXCELLENT: Model learned to play optimally!")
    print("  Expected Strong Performance vs Minimax")
elif test_acc_optimal >= 60:
    print("~ GOOD: Model learned basic strategy")
    print("  Expected Decent Performance vs Minimax")
elif test_acc_optimal >= 40:
    print("- FAIR: Model has some strategy")
    print("  Expected Weak Performance vs Minimax")
else:
    print("✗ POOR: Model barely learned anything")
    print("  Expected Very Weak Performance vs Minimax")

print()
print("="*70)
