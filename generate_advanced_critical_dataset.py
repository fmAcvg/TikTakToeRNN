import numpy as np

def generate_advanced_critical_situations():
    """
    Generate advanced critical TicTacToe situations for training:
    - Complex blocking scenarios
    - Multi-step winning scenarios
    """
    advanced_critical_data = []

    # Example: Advanced blocking situations
    advanced_blocking = [
        # Opponent about to win with multiple threats
        ([ 1, -1,  0, -1,  1,  0,  0,  0,  0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([ 0,  1, -1,  0, -1,  1,  0,  0,  0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]

    # Example: Advanced winning scenarios
    advanced_winning = [
        # Player can win in two steps
        ([ 1,  1,  0, -1, -1,  0,  0,  0,  0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        ([ 0,  1,  1, -1, -1,  0,  0,  0,  0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]

    # Combine all situations
    for board, target in advanced_blocking + advanced_winning:
        board = np.array(board)
        target = np.array(target)
        advanced_critical_data.append(np.concatenate([board, target]))

    advanced_critical_data = np.array(advanced_critical_data)
    np.save('datasets/tictactoe_advanced_critical.npy', advanced_critical_data)
    print(f"Generated {len(advanced_critical_data)} advanced critical situations.")

if __name__ == "__main__":
    generate_advanced_critical_situations()