#lenny
import numpy as np
from tqdm import tqdm


def check_winner(board):
    b = board.reshape(3, 3)
    lines = [
        b[0], b[1], b[2],
        b[:, 0], b[:, 1], b[:, 2],
        b.diagonal(), np.fliplr(b).diagonal()
    ]
    if any(np.all(line == 1) for line in lines):
        return 1
    if any(np.all(line == -1) for line in lines):
        return -1
    return 0


def get_next_boards(board, player):
    next_states = []
    for i in range(9):
        if board[i] == 0:
            new_board = board.copy()
            new_board[i] = player
            next_states.append(new_board)
    return next_states


def minimax(board, player, alpha=-999, beta=999):
    winner = check_winner(board)
    if winner != 0:
        return winner
    if np.all(board != 0):
        return 0
    if player == 1:
        max_eval = -999
        for next_board in get_next_boards(board, 1):
            eval = minimax(next_board, -1, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 999
        for next_board in get_next_boards(board, -1):
            eval = minimax(next_board, 1, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_current_player(board):
    """Bestimmt, wer am Zug ist: 1=X, -1=O (X beginnt)."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    return 1 if x_count == o_count else -1


def find_best_move(board):
    """Bester Zug für Spieler 1 (X)."""
    best_value = -999
    best_move = None
    for i in range(9):
        if board[i] == 0:
            new_board = board.copy()
            new_board[i] = 1
            value = minimax(new_board, -1)
            if value > best_value:
                best_value = value
                best_move = i
    return best_move


def find_best_move_for_player(board, player):
    """Bester Zug für den am Zug befindlichen Spieler (1=X, -1=O)."""
    if player == 1:
        return find_best_move(board)
    # Spieler 2 (O): minimierender Spieler
    best_value = 999
    best_move = None
    for i in range(9):
        if board[i] == 0:
            new_board = board.copy()
            new_board[i] = -1
            value = minimax(new_board, 1)
            if value < best_value:
                best_value = value
                best_move = i
    return best_move


def find_all_best_moves_for_player(board, player):
    """Alle optimalen Züge für den aktuellen Spieler (nicht nur den ersten Treffer)."""
    empty_positions = np.where(board == 0)[0]
    if len(empty_positions) == 0:
        return []

    move_scores = []
    for i in empty_positions:
        new_board = board.copy()
        new_board[i] = player
        next_player = -player
        score = minimax(new_board, next_player)
        move_scores.append((int(i), score))

    if player == 1:
        best_score = max(score for _, score in move_scores)
    else:
        best_score = min(score for _, score in move_scores)

    return [move for move, score in move_scores if score == best_score]


# -----------------------------
# Datensatzgenerator mit Limit und integriertem Label
# -----------------------------
def generate_dataset(max_boards=5000):
    data = []  # jedes Element: [b0, b1, ..., b8, best_move]

    start_board = np.zeros(9, dtype=int)
    states = [(start_board, 1)]  # Liste von (Board, player_to_move)

    pbar = tqdm(total=max_boards, desc="Boards generiert", unit="Boards")

    while states and len(data) < max_boards:
        board, player = states.pop()
        winner = check_winner(board)
        if winner != 0 or np.all(board != 0):
            continue

        best_move = find_best_move_for_player(board, player)
        data.append(np.append(board, best_move))
        pbar.update(1)

        for next_board in get_next_boards(board, player):
            states.append((next_board, -player))

    pbar.close()
    return np.array(data, dtype=int)


# -----------------------------
# Speichern
# -----------------------------
if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe Datensatz generieren")
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=1000,
        help="Anzahl der zu generierenden Boards (Standard: 1000). Beispiele: 1000, 250000"
    )
    args = parser.parse_args()
    
    dataset = generate_dataset(max_boards=args.size)
    num_samples = len(dataset)
    print("Datensatz fertig. Anzahl Boards:", num_samples)
    
    # In datasets-Ordner speichern
    datasets_dir = "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    filename = f"tictactoe_dataset_{num_samples}.npy"
    filepath = os.path.join(datasets_dir, filename)
    np.save(filepath, dataset)
    print(f"Dataset gespeichert als {filepath}")



#Minimax dataset generator komplett von chatgpt erstellt mit diesem Prompt:

'''„Erstelle ein vollständiges, eigenständiges Python-Skript zur Generierung eines TicTacToe-Datensatzes 
mithilfe des Minimax-Algorithmus.Das Spielfeld soll als NumPy-Array mit 9 Feldern dargestellt werden 
(1 = eigener Spieler, -1 = Gegner, 0 = leer).
Es sollen ausschließlich gültige Spielstände erzeugt werden.
Für jeden Spielstand soll mit Minimax inklusive Alpha-Beta-Pruning der beste nächste Zug für Spieler 1 
berechnet werden.Board und bester Zug sollen gemeinsam in einem NumPy-Array gespeichert werden.
Die Generierung soll nach einer festen Anzahl von Spielständen abbrechen und den Fortschritt live anzeigen.
Am Ende soll der Datensatz als .npy-Datei gespeichert werden.“'''


