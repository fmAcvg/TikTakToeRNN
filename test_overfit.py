import numpy as np
from NeuralNetwork.predict import NeuralNetwork, create_layer
from NeuralNetwork.training import train_on_data_set
import os

# Create 20 samples
data = np.load("datasets/tictactoe_dataset_5000_optimal.npy")[:20]

model = NeuralNetwork([create_layer(9, 64), create_layer(64, 64), create_layer(64, 9)])
print("Starting Overfit-20 Test...")

for epoch in range(100):
    losses = []
    corrects = []
    for sample in data:
        board = sample[:9].astype(float)
        target = sample[9:].astype(int)
        loss, correct = train_on_data_set(
            model, board, target,
            learning_rate=0.01,
            weight_decay=0.0,
            legal_move_mask=(board == 0).astype(float),
            multi_hot=True
        )
        losses.append(loss)
        corrects.append(correct)
    
    acc = np.mean(corrects) * 100
    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch}: Loss {np.mean(losses):.4f}, Acc {acc:.1f}%")

print("Finished!")
