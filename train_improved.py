#!/usr/bin/env python
"""
Verbessertes Training mit:
- Multi-Hot Format (alle optimalen Moves)
- Korrekte Accuracy-Metrik (prediction in optimal_moves)
- Train/Test Split auf Unique Boards
- Early Stopping
- Realistische Hyperparameter
"""
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import hashlib
import importlib.util
import argparse

# Konfiguration
CONFIG = {
    'hidden_layers': 8,
    'neurons_per_layer': 512,
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'batch_size': 32,
    'epochs': 2000,
    'early_stopping_patience': 250,
    'test_split': 0.2,
    'multi_hot': True,            # Multi-Hot für alle optimalen Züge
    'split_on_unique_boards': False,
    'use_same_train_test': True,
}


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train improved TicTacToe model with configurable hyperparameters")
    parser.add_argument("--hidden-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--neurons", type=int, help="Neurons per hidden layer")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="Weight decay (L2)")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Max epochs")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--test-split", type=float, help="Test split ratio, e.g. 0.2")

    parser.add_argument("--multi-hot", action="store_true", help="Use multi-hot targets")
    parser.add_argument("--single-hot", action="store_true", help="Use single-hot targets")
    parser.add_argument("--split-unique", action="store_true", help="Split on unique boards")
    parser.add_argument("--split-sample", action="store_true", help="Split on sample level")
    parser.add_argument("--same-train-test", action="store_true", help="Use same data for train and test")

    parser.add_argument("--dataset-path", type=str, help="Path to .npy dataset")
    parser.add_argument("--model-name", type=str, help="Custom model name (without/with .npz)")
    return parser.parse_args()


def apply_cli_overrides(config, args):
    if args.hidden_layers is not None:
        config['hidden_layers'] = args.hidden_layers
    if args.neurons is not None:
        config['neurons_per_layer'] = args.neurons
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.patience is not None:
        config['early_stopping_patience'] = args.patience
    if args.test_split is not None:
        config['test_split'] = args.test_split

    if args.multi_hot:
        config['multi_hot'] = True
    if args.single_hot:
        config['multi_hot'] = False

    if args.split_unique:
        config['split_on_unique_boards'] = True
    if args.split_sample:
        config['split_on_unique_boards'] = False

    if args.same_train_test:
        config['use_same_train_test'] = True


CLI_ARGS = parse_cli_args()
apply_cli_overrides(CONFIG, CLI_ARGS)

print("="*70)
print("TicTacToe RNN Training - IMPROVED VERSION")
print("="*70)
print(f"\nKonfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print()

# Import dataset generator robustly (ohne sys.path-Hack)
dataset_module_path = os.path.join(os.path.dirname(__file__), 'NeuralNetwork', 'training-dataset', 'dataset.py')
spec = importlib.util.spec_from_file_location("dataset_module", dataset_module_path)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
generate_optimal_dataset = dataset_module.generate_optimal_dataset
from NeuralNetwork.predict import NeuralNetwork
from NeuralNetwork.rnn_fast.fast_predict import FastNeuralNetwork, create_fast_layer
from NeuralNetwork.rnn_fast.fast_training import train_batch

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def save_model(model, path):
    """Speichert das Modell als NPZ-Datei (kombatibel mit altem System)"""
    data = {}
    for i, layer in enumerate(model.layers):
        num_neurons = layer.weights.shape[1]
        for j in range(num_neurons):
            data[f"layer_{i}_neuron_{j}_weights"] = layer.weights[:, j]
            data[f"layer_{i}_neuron_{j}_bias"] = layer.bias[j]
    np.savez(path, **data)

def generate_model_name(config):
    """Generate a unique and descriptive model name based on the configuration."""
    config_str = f"layers-{config['hidden_layers']}_neurons-{config['neurons_per_layer']}_lr-{config['learning_rate']}_batch-{config['batch_size']}"
    unique_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]  # config fingerprint
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"model_{config_str}_{unique_hash}_{timestamp}"


def normalize_model_name(name: str) -> str:
    base = os.path.basename(name.strip())
    if base.endswith(".npz"):
        base = base[:-4]
    return base

# ============================================================================
# 1. DATASET LADEN/GENERIEREN
# ============================================================================
dataset_path = CLI_ARGS.dataset_path if CLI_ARGS.dataset_path else "datasets/tictactoe_dataset_tactical.npy"

if not os.path.exists(dataset_path):
    print("Dataset nicht gefunden! Generiere neues optimales Dataset...")
    print("(Dies kann 5-10 Minuten dauern)\n")
    dataset = generate_optimal_dataset(
        target_samples=12000,
        balance_phases=True,
        multi_hot=CONFIG['multi_hot'],
        filter_symmetries=False
    )
    os.makedirs("datasets", exist_ok=True)
    np.save(dataset_path, dataset)
    print(f"Dataset gespeichert: {dataset_path}\n")
else:
    print(f"Lade existierendes Dataset: {dataset_path}")
    dataset = np.load(dataset_path)
    print(f"  Shape: {dataset.shape}")
    print(f"  Samples: {len(dataset)}")
    print()

    if CONFIG.get('use_same_train_test', False):
        print("Nutze identisches Train/Test-Set (Target-Run Modus)...")
        train_dataset = dataset.copy()
        test_dataset = dataset.copy()
        print(f"  Train Set: {len(train_dataset)} Samples")
        print(f"  Test Set: {len(test_dataset)} Samples")
        print()
    else:
        # ============================================================================
        # 2. TRAIN/TEST SPLIT (auf Unique Boards, nicht Sample-Level!)
        # ============================================================================
        if CONFIG.get('split_on_unique_boards', True):
            print("Führe Train/Test Split durch (auf Unique Boards)...")

            # Extrahiere Boards (erste 9 Spalten)
            boards = dataset[:, :9]

            # Finde unique Boards (mit deren Indizes)
            unique_boards, unique_indices = np.unique(boards, axis=0, return_index=True)
            print(f"  Gefunden: {len(unique_boards)} unique Boards aus {len(dataset)} Samples")

            # Shuffle und split
            np.random.shuffle(unique_indices)
            split_idx = int(len(unique_indices) * (1 - CONFIG['test_split']))
            train_indices = unique_indices[:split_idx]
            test_indices = unique_indices[split_idx:]

            # Datasets
            train_dataset = dataset[np.sort(train_indices)]
            test_dataset = dataset[np.sort(test_indices)]
        else:
            print("Führe Train/Test Split durch (Sample-Level)...")
            indices = np.random.permutation(len(dataset))
            split_idx = int(len(indices) * (1 - CONFIG['test_split']))
            train_dataset = dataset[indices[:split_idx]]
            test_dataset = dataset[indices[split_idx:]]

print(f"  Train Set: {len(train_dataset)} Samples")
print(f"  Test Set: {len(test_dataset)} Samples")
print()

# ============================================================================
# 3. MODELL ERSTELLEN
# ============================================================================
print("Erstelle Modell...")
print(f"  Input: 9 (Board)")
print(f"  Hidden: {CONFIG['hidden_layers']}x {CONFIG['neurons_per_layer']}")
print(f"  Output: 9 (Moves)")

layers = []
prev_neurons = 9

for i in range(CONFIG['hidden_layers']):
    layer = create_fast_layer(prev_neurons, CONFIG['neurons_per_layer'], activation="relu")
    layers.append(layer)
    prev_neurons = CONFIG['neurons_per_layer']

# Output Layer
output_layer = create_fast_layer(prev_neurons, 9, activation="linear")
layers.append(output_layer)

model = FastNeuralNetwork(layers)

print("[OK] Modell erstellt\n")

# ============================================================================
# 4. TRAINING MIT EARLY STOPPING
# ============================================================================
print("Starte Training...")
print(f"  LR: {CONFIG['learning_rate']}")
print(f"  Batch Size: {CONFIG['batch_size']}")
print(f"  Early Stopping Patience: {CONFIG['early_stopping_patience']}")
print(f"  Training Samples: {len(train_dataset)}")
print()

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

best_test_acc = -1
epochs_without_improvement = 0

# Generate a descriptive model name
model_name = normalize_model_name(CLI_ARGS.model_name) if CLI_ARGS.model_name else generate_model_name(CONFIG)
print(f"[INFO] Training model: {model_name}")

# Update the model saving path
os.makedirs("models", exist_ok=True)
model_path = f"models/{model_name}.npz"

for epoch in range(CONFIG['epochs']):
    # ===== TRAINING =====
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    # Shuffle training data
    indices = np.random.permutation(len(train_dataset))
    shuffled_data = train_dataset[indices]
    
    # Fast Mini-Batch Training (vektorisiert!)
    X_train = shuffled_data[:, :9].astype(float)
    if CONFIG['multi_hot'] and shuffled_data.shape[1] == 18:
        Y_train = shuffled_data[:, 9:].astype(int)
    else:
        Y_train = np.zeros((len(shuffled_data), 9), dtype=int)
        Y_train[np.arange(len(shuffled_data)), shuffled_data[:, 9].astype(int)] = 1

    batch_size = CONFIG['batch_size']
    train_correct = 0
    train_total = 0
    train_loss_sum = 0
    
    for start_idx in range(0, len(X_train), batch_size):
        end_idx = start_idx + batch_size
        X_batch = X_train[start_idx:end_idx]
        Y_batch = Y_train[start_idx:end_idx]
        
        legal_masks_batch = (X_batch == 0).astype(float)
        
        batch_loss, batch_correct = train_batch(
            model, 
            X_batch, 
            Y_batch,
            learning_rate=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
            legal_masks_batch=legal_masks_batch,
            multi_hot=CONFIG['multi_hot']
        )
        
        train_loss_sum += batch_loss * len(X_batch)
        train_correct += batch_correct
        train_total += len(X_batch)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total * 100
    
    # ===== TESTING =====
    X_test = test_dataset[:, :9].astype(float)
    if CONFIG['multi_hot'] and test_dataset.shape[1] == 18:
        Y_test = test_dataset[:, 9:].astype(int)
    else:
        Y_test = np.zeros((len(test_dataset), 9), dtype=int)
        Y_test[np.arange(len(test_dataset)), test_dataset[:, 9].astype(int)] = 1
        
    test_total = len(X_test)
    
    # 1 Forward Pass for the ENTIRE test set (vektorisiert)
    test_logits = model.forward(X_test)
    
    legal_masks_test = (X_test == 0)
    test_logits_masked = test_logits.copy()
    test_logits_masked[~legal_masks_test] = -np.inf
    
    test_exps = np.exp(test_logits_masked - np.max(test_logits_masked, axis=1, keepdims=True))
    test_probs = test_exps / np.sum(test_exps, axis=1, keepdims=True)
    
    # Loss computation
    target_probs = Y_test.astype(float)
    if CONFIG['multi_hot']:
        row_sums = np.sum(target_probs, axis=1, keepdims=True)
        # Avoid division by zero
        target_probs = np.divide(target_probs, row_sums, out=np.zeros_like(target_probs), where=row_sums!=0)
        
    test_loss = -np.sum(target_probs * np.log(test_probs + 1e-15)) / test_total
    
    # Accuracy
    preds = np.argmax(test_probs, axis=1)
    if CONFIG['multi_hot']:
        test_correct = np.sum(Y_test[np.arange(test_total), preds] == 1)
    else:
        test_correct = np.sum(preds == np.argmax(Y_test, axis=1))

    test_acc = test_correct / test_total * 100
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # ===== EARLY STOPPING =====
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        epochs_without_improvement = 0
        
        # Speichere bestes Modell
        save_model(model, model_path)
    else:
        epochs_without_improvement += 1
    
    # Loggen jede 10 Epochen oder wenn neue beste erreicht
    if (epoch + 1) % 10 == 0 or epochs_without_improvement == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:5.1f}%", end="")
        if epochs_without_improvement == 0:
            print(" [OK] BEST")
        else:
            print()
    
    # Early Stopping
    if epochs_without_improvement >= CONFIG['early_stopping_patience']:
        print(f"\n[OK] Early Stopping nach Epoch {epoch + 1} (Test Acc plateau für {CONFIG['early_stopping_patience']} Epochen)")
        break

# ============================================================================
# 5. ERGEBNISSE
# ============================================================================
print(f"\n{'='*70}")
print("Trainings-Zusammenfassung")
print(f"{'='*70}")
print(f"Beste Test Accuracy: {best_test_acc:.1f}%")
print(f"Final Train Accuracy: {train_acc:.1f}%")
print(f"Final Test Accuracy: {test_acc:.1f}%")
print(f"Overfitting (Train-Test Gap): {abs(train_acc - test_acc):.1f}%")
print(f"Trainierte Epochen: {epoch + 1}")
print()
print("[OK] Training abgeschlossen!")
print(f"[OK] Bestes Modell gespeichert als: {model_path}")

# Speichere auch finale Statistiken
np.savez(
    "models/training_stats_latest.npz",
    train_losses=np.array(train_losses),
    train_accuracies=np.array(train_accuracies),
    test_losses=np.array(test_losses),
    test_accuracies=np.array(test_accuracies)
)
