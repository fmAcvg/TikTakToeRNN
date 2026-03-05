"""
Neue Accuracy-Metrik: prediction ∈ optimal_moves statt prediction == label
"""
import numpy as np
import sys
import importlib.util
from main import load_model

sys.path.append('NeuralNetwork/training-dataset')
from dataset import canonicalize_board, find_all_best_moves_for_player

spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)


def compute_fair_accuracy(model, dataset_path, num_samples=1000):
    """
    Berechnet echte Accuracy: prediction ∈ optimal_moves
    
    Nicht: prediction == label (der eine optimale Zug im Dataset)
    Sondern: prediction ist einer der vielen optimalen Züge (Minimax sagt alle)
    """
    
    # Lade Dataset
    data = np.load(dataset_path)
    boards = data[:, :9]
    moves = data[:, 9]
    
    if num_samples > 0:
        indices = np.random.choice(len(boards), min(num_samples, len(boards)), replace=False)
        boards = boards[indices]
        moves = moves[indices]
    
    correct = 0
    total = 0
    
    for i in range(len(boards)):
        board = boards[i]
        label = int(moves[i])
        
        # Bestimme player
        num_stones = np.sum(board != 0)
        player = 1 if num_stones % 2 == 0 else -1
        
        # Alle optimalen Züge für diese Position
        all_optimal = dataset_module.find_all_best_moves_for_player(board, player)
        
        if not all_optimal:
            continue
        
        # Kanonisierung
        canonical_board = canonicalize_board(board, player)
        legal_mask = (canonical_board == 0).astype(float)
        
        # Modell-Vorhersage
        pred, _ = model.predict(canonical_board.astype(float), legal_mask)
        
        # Neue Metrik: Ist Prediction in allen optimalen Zügen?
        if pred in all_optimal:
            correct += 1
        
        total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, total, correct


if __name__ == "__main__":
    print("="*70)
    print("Neue Accuracy-Metrik: prediction ∈ optimal_moves")
    print("="*70)
    print()
    
    # Lade Modell
    model_path = "models/model_2026-03-04T20-22-04.npz"
    model = load_model(model_path)
    
    if model is None:
        print(f"Fehler: Modell nicht geladen!")
        sys.exit(1)
    
    print(f"Modell: {model_path}\n")
    
    # Test auf verschiedenen Datasets
    datasets = [
        ("datasets/tictactoe_dataset_1000.npy", 1000),
        ("datasets/tictactoe_dataset_7500.npy", 500),
        ("datasets/tictactoe_dataset_15000.npy", 500),
    ]
    
    for dataset_path, num_samples in datasets:
        try:
            print(f"Teste auf {dataset_path}...")
            acc, total, correct = compute_fair_accuracy(model, dataset_path, num_samples)
            print(f"  Accuracy (fair): {acc:.1f}% ({correct}/{total})")
            print()
        except Exception as e:
            print(f"  Fehler: {e}\n")
    
    print("="*70)
    print("FAZIT:")
    print("  Fair Accuracy sollte 80-95% sein (nicht 50-60% wie alte Metrik)")
    print("  Das zeigt dass Modell oft richtig spielt, aber andere")
    print("  gleichwertige Züge wählt als der eine im Dataset.")
    print("="*70)
