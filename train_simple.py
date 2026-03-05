#!/usr/bin/env python3
"""
Simple, focused TicTacToe RNN Training
- Minimal parameter count (~400 total)
- Fast training (50 epochs)
- Minimax-quality opponent
- No over-engineering
"""

import numpy as np
import os
from datetime import datetime
import sys

sys.path.insert(0, 'NeuralNetwork/training-dataset')
from dataset import (
    generate_all_legal_boards,
    is_canonical,
    canonicalize_board,
    get_current_player,
    find_all_best_moves_for_player
)

from NeuralNetwork.predict import NeuralNetwork, create_layer


def generate_dataset_simple(n_samples=500):
    """Generate a focused dataset for quick training"""
    print("Generating dataset...")
    
    # Generate all canonical boards
    all_boards = generate_all_legal_boards()
    canonical_boards = [b for b in all_boards if is_canonical(b)]
    
    data = []
    for board in canonical_boards:
        player = get_current_player(board)
        optimal_moves = find_all_best_moves_for_player(board, player)
        
        if not optimal_moves:
            continue
        
        canonical_board = canonicalize_board(board, player)
        
        # Create one-hot label for best move (just take first best move)
        label = np.zeros(9, dtype=int)
        label[optimal_moves[0]] = 1
        
        data.append(np.concatenate([canonical_board.astype(int), label]))
    
    # Convert to numpy and take random sample
    data = np.array(data, dtype=int)
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        data = data[indices]
    
    print(f"✓ Generated {len(data)} unique boards")
    
    # Save
    os.makedirs("datasets", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filepath = f"datasets/tictactoe_simple_{len(data)}_{timestamp}.npy"
    np.save(filepath, data)
    print(f"✓ Saved to {filepath}")
    
    return filepath


def train_simple():
    """Train a simple, focused model"""
    
    # CONFIG
    CONFIG = {
        'epochs': 50,
        'learning_rate': 0.005,
        'weight_decay': 0.0,
        'batch_size': 1,  # SGD
        'patience': 15,
        'hidden_size': 32,  # 9→32→9 = ~400 params
    }
    
    print("\n" + "="*70)
    print("TicTacToe RNN - Simple Training")
    print("="*70)
    print(f"\nConfig: {CONFIG}\n")
    
    # === LOAD/GENERATE DATASET ===
    dataset_dir = "datasets"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Try to find existing dataset
    datasets = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npy')], 
                     key=lambda x: os.path.getctime(os.path.join(dataset_dir, x)), 
                     reverse=True)
    
    if datasets:
        dataset_path = os.path.join(dataset_dir, datasets[0])
        print(f"✓ Using dataset: {datasets[0]}")
    else:
        print("⚠ No dataset found, generating...")
        dataset_path = generate_dataset_simple(500)
    
    # Load dataset
    data = np.load(dataset_path)
    boards = data[:, :9].astype(float)
    labels = data[:, 9:].astype(float)
    
    print(f"✓ Loaded {len(boards)} samples")
    
    # Train/Test split
    n = len(boards)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    train_boards = boards[train_idx]
    train_labels = labels[train_idx]
    test_boards = boards[test_idx]
    test_labels = labels[test_idx]
    
    print(f"✓ Train: {len(train_boards)}, Test: {len(test_boards)}\n")
    
    # === CREATE MODEL ===
    # Simple: 9 → 32 → 9
    # Parameters: 9*32 + 32 + 32*9 + 9 = 288 + 32 + 288 + 9 = 617 (acceptable)
    layers = [
        create_layer(9, CONFIG['hidden_size']),      # 9→32
        create_layer(CONFIG['hidden_size'], 9),      # 32→9
    ]
    model = NeuralNetwork(layers)
    
    print(f"✓ Model: 9 → {CONFIG['hidden_size']} → 9")
    print(f"✓ Parameters: ~{9*CONFIG['hidden_size'] + CONFIG['hidden_size']*9 + CONFIG['hidden_size'] + 9}\n")
    
    # === TRAINING LOOP ===
    print("Training...")
    print("-" * 70)
    
    best_test_acc = 0
    no_improve = 0
    
    for epoch in range(CONFIG['epochs']):
        # === TRAIN ===
        train_loss = 0.0
        train_correct = 0
        
        # Shuffle train indices
        train_perm = np.random.permutation(len(train_boards))
        
        for idx in train_perm:
            board = train_boards[idx]
            target = train_labels[idx]
            
            # Forward pass
            activation = board.copy()
            for layer in model.layers[:-1]:
                activation = layer.forward(activation)
            
            # Output layer (linear)
            output_layer = model.layers[-1]
            logits = np.array([np.dot(n.weights, activation) + n.bias 
                             for n in output_layer.neurons])
            
            # Softmax
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            
            # Loss & correct
            loss = -np.sum(target * np.log(probs + 1e-15))
            train_loss += loss
            
            if np.argmax(probs) == np.argmax(target):
                train_correct += 1
            
            # Backprop
            grad = probs - target
            
            # Output layer gradient
            new_grad = np.zeros_like(activation)
            for j, neuron in enumerate(output_layer.neurons):
                delta = grad[j]
                neuron.weights -= CONFIG['learning_rate'] * delta * activation
                neuron.bias -= CONFIG['learning_rate'] * delta
                new_grad += delta * neuron.weights
            grad = new_grad
            
            # Hidden layer gradient (tanh)
            for i in range(len(model.layers) - 2, -1, -1):
                layer = model.layers[i]
                layer_input = train_boards[idx] if i == 0 else activations[i-1]
                
                new_grad = np.zeros_like(layer_input)
                for j, neuron in enumerate(layer.neurons):
                    tanh_grad = 1 - neuron.output ** 2  # tanh derivative
                    delta = grad[j] * tanh_grad
                    neuron.weights -= CONFIG['learning_rate'] * delta * layer_input
                    neuron.bias -= CONFIG['learning_rate'] * delta
                    new_grad += delta * neuron.weights
                grad = new_grad
        
        avg_train_loss = train_loss / len(train_boards)
        train_acc = 100.0 * train_correct / len(train_boards)
        
        # === TEST ===
        test_loss = 0.0
        test_correct = 0
        
        for idx in range(len(test_boards)):
            board = test_boards[idx]
            target = test_labels[idx]
            
            # Forward
            activation = board.copy()
            for layer in model.layers[:-1]:
                activation = layer.forward(activation)
            
            output_layer = model.layers[-1]
            logits = np.array([np.dot(n.weights, activation) + n.bias 
                             for n in output_layer.neurons])
            
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            
            loss = -np.sum(target * np.log(probs + 1e-15))
            test_loss += loss
            
            if np.argmax(probs) == np.argmax(target):
                test_correct += 1
        
        avg_test_loss = test_loss / len(test_boards)
        test_acc = 100.0 * test_correct / len(test_boards)
        
        # === EARLY STOPPING ===
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improve = 0
            
            # Save best model
            os.makedirs("models", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_path = f"models/simple_best_{timestamp}.npz"
            
            weights = {}
            for i, layer in enumerate(model.layers):
                for j, neuron in enumerate(layer.neurons):
                    weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                    weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
            
            np.savez(save_path, **weights)
        else:
            no_improve += 1
        
        # === PRINT PROGRESS ===
        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = " ✓ BEST" if test_acc > best_test_acc else ""
            print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:6.2f}% {marker}")
        
        # Early stopping
        if no_improve >= CONFIG['patience']:
            print(f"\n⊘ Early stopping at epoch {epoch+1}")
            break
    
    # === FINAL ===
    print("-" * 70)
    print(f"\n✓ Training complete!")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"  Model saved to: models/simple_best_*.npz")
    print()


if __name__ == "__main__":
    train_simple()
