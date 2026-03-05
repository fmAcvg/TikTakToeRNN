import numpy as np

def generate_critical_situations():
    """
    Generate critical TicTacToe situations for training:
    - Blocking opponent's winning move
    - Making a winning move
    """
    critical_data = []

    # Example: Blocking situations
    blocking_situations = [
        # Opponent about to win horizontally
        ([-1, -1, 0,  1,  1,  0,  0,  0,  0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        # Opponent about to win vertically
        ([ 1,  0,  0, -1,  0,  0, -1,  0,  0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),
        # Opponent about to win diagonally
        ([ 1,  0,  0,  0, -1,  0,  0,  0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ]

    # Example: Winning situations
    winning_situations = [
        # Player about to win horizontally
        ([ 1,  1,  0, -1, -1,  0,  0,  0,  0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        # Player about to win vertically
        ([ 1,  0,  0,  1,  0,  0, -1,  0,  0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),
        # Player about to win diagonally
        ([ 1,  0,  0,  0,  1,  0,  0,  0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ]

    # Combine all situations
    for board, target in blocking_situations + winning_situations:
        board = np.array(board)
        target = np.array(target)
        critical_data.append(np.concatenate([board, target]))

    critical_data = np.array(critical_data)
    np.save('datasets/tictactoe_critical.npy', critical_data)
    print(f"Generated {len(critical_data)} critical situations.")

if __name__ == "__main__":
    generate_critical_situations()