import numpy as np
from tqdm import tqdm


# -----------------------------
# Minimax-Funktionen
# -----------------------------
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


def find_best_move(board):
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

        best_move = find_best_move(board)
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
    MAX_BOARDS = 5000
    dataset = generate_dataset(MAX_BOARDS)
    print("Datensatz fertig. Anzahl Boards:", len(dataset))
    np.save("tictactoe_dataset.npy", dataset)
    print("Dataset gespeichert als tictactoe_dataset.npy")
