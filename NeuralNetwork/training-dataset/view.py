import numpy as np

def load_dataset(filename="tictactoe_dataset.npy"):
    """
    Lädt das Dataset aus einer .npy Datei.
    Jede Zeile: [b0,b1,...,b8,best_move]
    """
    return np.load(filename)

def display_board(board):
    """
    Zeigt ein Board als 3x3 Matrix.
    1  -> X
    -1 -> O
    0  -> leer
    """
    symbols = {1: "X", -1: "O", 0: " "}
    for i in range(3):
        row = [symbols[val] for val in board[i*3:(i+1)*3]]
        print(" | ".join(row))
        if i < 2:
            print("-"*5)
    print()

def interactive_viewer(dataset):
    """
    Interaktive Ansicht: Nächstes/prev Board durch Tasten
    q -> quit
    n -> next
    p -> previous
    """
    idx = 0
    while True:
        print(f"\nBoard {idx+1}/{len(dataset)}:")
        display_board(dataset[idx,:9])
        print("Best Move (Label):", dataset[idx,9])

        cmd = input("Befehl [n=next, p=prev, q=quit]: ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "n":
            if idx < len(dataset)-1:
                idx += 1
            else:
                print("Letztes Board erreicht.")
        elif cmd == "p":
            if idx > 0:
                idx -= 1
            else:
                print("Erstes Board erreicht.")
        else:
            print("Ungültiger Befehl!")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    filename = "tictactoe_dataset.npy"  # dein Dataset
    dataset = load_dataset(filename)
    print(f"Dataset geladen: {dataset.shape[0]} Boards")
    interactive_viewer(dataset)
