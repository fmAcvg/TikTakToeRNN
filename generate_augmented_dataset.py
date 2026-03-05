#!/usr/bin/env python
"""
Dataset generation with proper augmentation using symmetries
"""
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, 'NeuralNetwork/training-dataset')
from dataset import (
    generate_all_legal_boards, 
    is_canonical, 
    canonicalize_board,
    get_current_player,
    find_all_best_moves_for_player
)

def augment_dataset_with_symmetries(canonical_dataset):
    """
    Augmentiert Datensatz durch Anwendung aller 8 Symmetrien.
    
    TicTacToe hat 8 Symmetrien (4 Rotationen × 2 Spiegelungen):
    - 4 Rotationen (0°, 90°, 180°, 270°)
    - 2 Spiegelungen (horizontal, vertikal)
    
    Diese erzeugen aus 799 unique Boards 799×8=6392 Augmentierte Samples!
    """
    symmetries = [
        # Rotation 0°
        lambda b: b,
        # Rotation 90° (clockwise)
        lambda b: np.array([b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]),
        # Rotation 180°
        lambda b: np.array([b[8], b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]]),
        # Rotation 270°
        lambda b: np.array([b[2], b[5], b[8], b[1], b[4], b[7], b[0], b[3], b[6]]),
        # Flip horizontal
        lambda b: np.array([b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]),
        # Flip vertical
        lambda b: np.array([b[6], b[7], b[8], b[3], b[4], b[5], b[0], b[1], b[2]]),
        # Flip diagonal (top-left to bottom-right)
        lambda b: np.array([b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]]),
        # Flip anti-diagonal
        lambda b: np.array([b[8], b[5], b[2], b[7], b[4], b[1], b[6], b[3], b[0]]),
    ]
    
    augmented_data = []
    move_mappings = [
        # Board positions after transformation
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0°
        [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 90° clockwise
        [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180°
        [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 270°
        [2, 1, 0, 5, 4, 3, 8, 7, 6],  # Horizontal flip
        [6, 7, 8, 3, 4, 5, 0, 1, 2],  # Vertical flip
        [0, 3, 6, 1, 4, 7, 2, 5, 8],  # Diagonal flip
        [8, 5, 2, 7, 4, 1, 6, 3, 0],  # Anti-diagonal flip
    ]
    
    for sample in canonical_dataset:
        board = sample[:9].astype(int)
        targets_orig = sample[9:].astype(int)
        
        for sym_idx, (sym_func, move_map) in enumerate(zip(symmetries, move_mappings)):
            aug_board = sym_func(board)
            
            # Transform move positions
            aug_targets = np.zeros(9, dtype=int)
            for orig_move in range(9):
                if targets_orig[orig_move] == 1:
                    new_move = move_map[orig_move]
                    aug_targets[new_move] = 1
            
            augmented_data.append(np.concatenate([aug_board, aug_targets]))
    
    return np.array(augmented_data, dtype=int)


# Generate base dataset (799 unique boards)
print("="*70)
print("Generiere Basis-Datensatz mit Symmetrie-Augmentation")
print("="*70)

all_boards = generate_all_legal_boards()
canonical_boards = [board for board in all_boards if is_canonical(board)]

print(f"Gefunden: {len(canonical_boards)} kanonische Boards")

# Generate multi-hot labels
data = []
for board in tqdm(canonical_boards, desc="Berechne optimale Moves"):
    player = get_current_player(board)
    optimal_moves = find_all_best_moves_for_player(board, player)
    
    if not optimal_moves:
        continue
    
    canonical_board = canonicalize_board(board, player)
    label = np.zeros(9, dtype=int)
    for move in optimal_moves:
        label[move] = 1
    
    data.append(np.concatenate([canonical_board, label]))

base_dataset = np.array(data, dtype=int)
print(f"\nBasis-Datensatz: {len(base_dataset)} Samples")

# Augment with symmetries
print("\nAugmentiere mit Symmetrien (8x)...")
augmented_dataset = augment_dataset_with_symmetries(base_dataset)
print(f"Augmentierter Datensatz: {len(augmented_dataset)} Samples")

# Save both
np.save("datasets/tictactoe_dataset_augmented.npy", augmented_dataset)
print(f"\nGespeichert: datasets/tictactoe_dataset_augmented.npy")

print(f"\n{'='*70}")
print(f"Datensatz-Statistik:")
print(f"  - Kanonische Boards: {len(base_dataset)}")
print(f"  - Nach 8x Symmetrie-Augmentation: {len(augmented_dataset)}")
print(f"  - Augmentationsfaktor: {len(augmented_dataset) / len(base_dataset):.1f}x")
print(f"{'='*70}")
