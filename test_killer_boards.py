import numpy as np
from main import load_model

def get_best_move(model, board, player):
    canonical = board * player
    legal_mask = (canonical == 0).astype(float)
    m, _ = model.predict(canonical.astype(float), legal_mask)
    return m

model_path = sorted([m for m in __import__("os").listdir("models/") if m.startswith("model_improved_")])[-1]
print("Testing:", model_path)
model = load_model("models/" + model_path)

b1 = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0])
m1 = get_best_move(model, b1, 1)

b2 = np.array([1, 0, 0, 0, -1, -1, 0, 0, 1])
m2 = get_best_move(model, b2, 1)

print(f"Win in 1: Move = {m1}, expected=2")
print(f"Block in 1: Move = {m2}, expected=3")
