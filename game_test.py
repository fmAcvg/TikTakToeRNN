"""
Test-Framework für Modell vs Minimax, Random, Model
"""
import numpy as np
import sys
import importlib.util
from main import load_model

# Lade Dataset-Modul
spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)

check_winner = dataset_module.check_winner
minimax = dataset_module.minimax
canonicalize_board = dataset_module.canonicalize_board
get_current_player = dataset_module.get_current_player
find_all_best_moves_for_player = dataset_module.find_all_best_moves_for_player


class Game:
    """TicTacToe Spiel-Engine"""
    
    def __init__(self, player1, player2, verbose=False):
        """
        player1, player2: Funktionen die (board, player) → move zurückgeben
        """
        self.board = np.zeros(9, dtype=int)
        self.player1 = player1  # 1 = X
        self.player2 = player2  # -1 = O
        self.verbose = verbose
    
    def play(self):
        """Spielt ein vollständiges Spiel. Returns: 1 (X wins), -1 (O wins), 0 (Draw)"""
        current_player = 1
        while True:
            # Spiel-Ende prüfen
            winner = check_winner(self.board)
            if winner != 0:
                if self.verbose:
                    print(f"Spieler {winner} gewinnt!")
                return winner
            
            if np.all(self.board != 0):
                if self.verbose:
                    print("Draw!")
                return 0
            
            # Zug
            player_func = self.player1 if current_player == 1 else self.player2
            try:
                move = player_func(self.board.copy(), current_player)
            except Exception as e:
                if self.verbose:
                    print(f"Fehler bei Spieler {current_player}: {e}")
                return -current_player  # Gegner gewinnt
            
            if move is None or move < 0 or move > 8 or self.board[move] != 0:
                if self.verbose:
                    print(f"Illegaler Zug von Spieler {current_player}: {move}")
                return -current_player  # Gegner gewinnt
            
            self.board[move] = current_player
            current_player = -current_player


def minimax_player(board, player):
    """Spielt optimal mit Minimax"""
    if player == 1:
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
    else:
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


def random_player(board, player):
    """Spielt zufällig"""
    legal_moves = np.where(board == 0)[0]
    if len(legal_moves) == 0:
        return None
    return np.random.choice(legal_moves)


def nn_player(model):
    """Erstellt einen NN-Player"""
    def play(board, player):
        # Kanonisierung
        canonical_board = canonicalize_board(board, player)
        
        # Legal mask
        legal_mask = (canonical_board == 0).astype(float)
        
        # Prediction
        move, _ = model.predict(canonical_board.astype(float), legal_mask)
        
        # Debugging
        # print(f"NN: board={board}, player={player}, move={move}")
        
        return int(move)
    
    return play


def run_games(player1_name, player1_func, player2_name, player2_func, num_games=100):
    """Spielt num_games Spiele und zählt Ergebnisse"""
    p1_wins = 0  # player1
    p2_wins = 0  # player2
    draws = 0
    
    for i in range(num_games):
        game = Game(player1_func, player2_func, verbose=False)
        result = game.play()
        
        if result == 1:
            p1_wins += 1
        elif result == -1:
            p2_wins += 1
        else:
            draws += 1
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_games} Spiele: {player1_name} {p1_wins}-{draws}-{p2_wins} {player2_name}")
    
    print(f"\n{'='*60}")
    print(f"{player1_name} vs {player2_name}: {num_games} Spiele")
    print(f"  {player1_name} Wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"  {player2_name} Wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return p1_wins, draws, p2_wins


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TicTacToe Modell-Evaluierung")
    print("="*60 + "\n")
    
    # Lade Modell
    model_path = "models/model_2026-03-04T20-22-04.npz"
    model = load_model(model_path)
    
    if model is None:
        print(f"Fehler: Modell {model_path} nicht geladen!")
        sys.exit(1)
    
    print(f"Modell geladen: {model_path}\n")
    
    nn = nn_player(model)
    
    # Test 1: NN vs Minimax (NN first)
    print("Test 1: NN vs Minimax (NN spielt X)")
    run_games("NN", nn, "Minimax", minimax_player, 100)
    
    # Test 2: NN vs Minimax (NN second)
    print("Test 2: NN vs Minimax (NN spielt O)")
    run_games("Minimax", minimax_player, "NN", nn, 100)
    
    # Test 3: NN vs Random
    print("Test 3: NN vs Random (NN spielt X)")
    run_games("NN", nn, "Random", random_player, 100)
    
    # Test 4: Random vs NN
    print("Test 4: Random vs NN (NN spielt O)")
    run_games("Random", random_player, "NN", nn, 100)
    
    print("\n" + "="*60)
    print("FAZIT:")
    print("  Gegen Minimax: Erwartet ~100% Draw")
    print("  Gegen Random: Erwartet 90%+ Win")
    print("="*60 + "\n")
