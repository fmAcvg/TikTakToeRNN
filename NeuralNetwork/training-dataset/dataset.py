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


def find_immediate_winning_moves(board, player):
    """Gibt alle Züge zurück, die sofort gewinnen."""
    wins = []
    for i in np.where(board == 0)[0]:
        b2 = board.copy()
        b2[i] = player
        if check_winner(b2) == player:
            wins.append(int(i))
    return wins


def find_tactical_best_moves_for_player(board, player):
    """
    Taktische Priorisierung für robustere Labels:
    1) Sofort gewinnen, wenn möglich
    2) Sonst sofortigen Gegner-Gewinn blocken
    3) Sonst normale Minimax-Optimalzüge
    """
    # 1) Immediate Win
    winning_moves = find_immediate_winning_moves(board, player)
    if winning_moves:
        return winning_moves

    # 2) Immediate Block
    opp_winning_moves = find_immediate_winning_moves(board, -player)
    if opp_winning_moves:
        # Blocken = genau diese Felder besetzen
        blocking_moves = [m for m in opp_winning_moves if board[m] == 0]
        if blocking_moves:
            return blocking_moves

    # 3) Fallback: Minimax-best
    return find_all_best_moves_for_player(board, player)


# Sanity-Check Funktionen
def is_valid_board(board):
    """Prüft, ob ein Board einen validen Spielzustand repräsentiert."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    
    # X startet, also #X == #O oder #X == #O + 1
    if not (x_count == o_count or x_count == o_count + 1):
        return False
    
    # Max eine Person kann gewinnen
    winner = check_winner(board)
    if winner != 0:
        # Wenn X gewinnt, darf das Spiel nicht weitergegangen sein (O spielt nicht danach)
        if winner == 1:
            return o_count == x_count - 1
        # Wenn O gewinnt, darf das Spiel nicht weitergegangen sein (X spielt nicht danach)
        else:
            return x_count == o_count
    
    return True


def get_player_to_move(board):
    """Bestimmt wer am Zug ist und prüft ob das Board valid ist."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    
    if x_count == o_count:
        return 1  # X ist am Zug
    elif x_count == o_count + 1:
        return -1  # O ist am Zug
    else:
        return None  # Invalides Board


def canonicalize_board(board, player_to_move):
    """
    Transformiert das Board aus Sicht des spielenden Spielers.
    
    Ergebnis: Eigene Steine sind immer +1, Gegner immer -1.
    Dies ermöglicht es dem Netz, nicht "raten" zu müssen, wer dran ist.
    """
    return board * player_to_move


def get_legal_move_mask(board):
    """Gibt eine Maske (9er-Vektor) mit Indizes der legalen Züge."""
    return (board == 0).astype(float)


# -----------------------------
# Datensatzgenerator mit Limit und integriertem Label
# -----------------------------
def generate_dataset(max_boards=5000, mode="tree"):
    """Erzeugt einen Minimax-Datensatz mit Sanity-Checks und kanonisierten Boards.

    Zwei Modi stehen zur Verfügung:
    - "tree" (Standard): vollständige Tiefensuche wie zuvor. Geht den Spielbaum in
      Tiefenorder durch und sammelt bis zu *max_boards* Zustände.
    - "selfplay": simuliert Partien, wobei in jeder Position ein zufälliger optimaler
      Zug gewählt wird. Das ergibt realistische Verläufe, da frühe Spielzustände
      häufiger auftreten als späte. Besonders hilfreich, um das Modell auf echte
      Spiele vorzubereiten.

    Wichtige Eigenschaften:
    1. Boards werden kanonisiert: eigene Steine = +1, Gegner = -1 (aus Sicht des Spielers)
    2. Illegale/invalide Samples werden herausgefiltert
    3. Labels sind garantiert auf leeren Feldern
    4. Statistiken werden ausgegeben (Anteil verworfener Samples)

    Returns
    -------
    np.ndarray
        Array mit Form (N,10), bestehend aus kanonisiertem Board (9 Felder) + best_move (Index 0-8).
    """
    if mode == "tree":
        data = []
        invalid_count = 0
        illegal_label_count = 0

        start_board = np.zeros(9, dtype=int)
        states = [(start_board, 1)]  # Liste von (Board, player_to_move)

        pbar = tqdm(total=max_boards, desc="Boards generiert", unit="Boards")

        while states and len(data) < max_boards:
            board, player = states.pop()
            
            # Sanity-Check: Ist das Board valide?
            if not is_valid_board(board):
                invalid_count += 1
                continue
            
            winner = check_winner(board)
            if winner != 0 or np.all(board != 0):
                continue

            best_move = find_best_move_for_player(board, player)
            
            # Sanity-Check: Ist der Label legal?
            if best_move is None or board[best_move] != 0:
                illegal_label_count += 1
                continue
            
            # Kanonisierung: Board aus Sicht des spielenden Spielers
            canonical_board = canonicalize_board(board, player)
            
            # Speichern: [canonical_board (9 Felder), best_move (Index 0-8)]
            data.append(np.append(canonical_board, best_move))
            pbar.update(1)

            for next_board in get_next_boards(board, player):
                states.append((next_board, -player))

        pbar.close()
        
        # Statistiken
        total_seen = len(data) + invalid_count + illegal_label_count
        print(f"\n=== Tree Mode Statistiken ===")
        print(f"Gültige Samples: {len(data)}")
        print(f"Invalide Boards: {invalid_count} ({100*invalid_count/max(1,total_seen):.2f}%)")
        print(f"Illegale Labels: {illegal_label_count} ({100*illegal_label_count/max(1,total_seen):.2f}%)")
        
        return np.array(data, dtype=int)
    
    elif mode == "selfplay":
        data = []
        invalid_count = 0
        illegal_label_count = 0
        
        pbar = tqdm(total=max_boards, desc="Boards (selfplay)", unit="Boards")
        rng = np.random.default_rng()
        
        while len(data) < max_boards:
            board = np.zeros(9, dtype=int)
            player = 1
            
            # Simuliere ein Spiel, speichere alle besuchten Zustände
            while True:
                winner = check_winner(board)
                if winner != 0 or np.all(board != 0):
                    break
                
                # Sanity-Check: Board valide?
                if not is_valid_board(board):
                    invalid_count += 1
                    break
                
                best_moves = find_all_best_moves_for_player(board, player)
                if not best_moves:
                    break
                
                move = rng.choice(best_moves)
                
                # Sanity-Check: Label legal?
                if board[move] != 0:
                    illegal_label_count += 1
                    break
                
                # Kanonisierung: Board aus Sicht des spielenden Spielers
                canonical_board = canonicalize_board(board, player)
                
                # Speichern: [canonical_board (9 Felder), best_move (Index 0-8)]
                data.append(np.append(canonical_board, move))
                pbar.update(1)
                
                if len(data) >= max_boards:
                    break
                
                board = board.copy()
                board[move] = player
                player = -player
        
        pbar.close()
        
        # Statistiken
        total_seen = len(data) + invalid_count + illegal_label_count
        print(f"\n=== Selfplay Mode Statistiken ===")
        print(f"Gültige Samples: {len(data)}")
        print(f"Invalide Boards: {invalid_count} ({100*invalid_count/max(1,total_seen):.2f}%)")
        print(f"Illegale Labels: {illegal_label_count} ({100*illegal_label_count/max(1,total_seen):.2f}%)")
        
        return np.array(data, dtype=int)
    
    else:
        raise ValueError(f"Unbekannter Modus '{mode}'")


# -----------------------------
# NEUER OPTIMIERTER GENERATOR - BESSERE VERTEILUNG
# -----------------------------
def get_board_symmetries(board):
    """Gibt alle 8 symmetrischen Varianten eines Boards zurück."""
    b = board.reshape(3, 3)
    symmetries = []
    
    # Original
    symmetries.append(board.copy())
    
    # 90° Rotation
    symmetries.append(np.rot90(b, 1).flatten())
    
    # 180° Rotation
    symmetries.append(np.rot90(b, 2).flatten())
    
    # 270° Rotation
    symmetries.append(np.rot90(b, 3).flatten())
    
    # Horizontal flip
    symmetries.append(np.fliplr(b).flatten())
    
    # Vertikal flip
    symmetries.append(np.flipud(b).flatten())
    
    # Diagonal flip (Hauptdiagonale)
    symmetries.append(b.T.flatten())
    
    # Anti-diagonal flip
    symmetries.append(np.fliplr(b.T).flatten())
    
    return symmetries


def is_canonical(board):
    """Prüft ob ein Board die kanonische Form ist (kleinste lexikografische Ordnung)."""
    symmetries = get_board_symmetries(board)
    canonical = min(symmetries, key=lambda x: tuple(x))
    return np.array_equal(board, canonical)


def generate_all_legal_boards():
    """Generiert alle legalen TicTacToe Boards systematisch."""
    legal_boards = []
    
    # Generiere alle möglichen Boards (3^9 = 19683)
    for i in range(3**9):
        # Konvertiere zu Basis-3
        board = []
        temp = i
        for _ in range(9):
            board.append(temp % 3 - 1)  # -1, 0, 1
            temp //= 3
        board = np.array(board, dtype=int)
        
        # Prüfe ob legal
        if is_valid_board(board):
            legal_boards.append(board)
    
    return legal_boards


def generate_optimal_dataset(target_samples=5000, balance_phases=True, multi_hot=True, filter_symmetries=False):
    """
    Generiert einen optimalen TicTacToe-Datensatz mit:
    - Alle legalen Boards (systematisch)
    - Symmetrien gefiltert (reduziert Duplikate)
    - Ausgewogene Verteilung über Spielphasen
    - MULTI-HOT Format: Alle optimalen Moves gespeichert (nicht nur einer!)
    
    Args:
        target_samples: Anzahl der Samples
        balance_phases: Ausgewogene Verteilung über Spielphasen
        multi_hot: Wenn True: 9-dim target (Multi-Hot), sonst nur erster Move
    
    Returns: 
        Wenn multi_hot=True: np.array mit Shape (N, 18) - [board(9), optimal_moves_mask(9)]
        Wenn multi_hot=False: np.array mit Shape (N, 10) - [board(9), first_optimal_move(1)]
    """
    print("Generiere alle legalen Boards...")
    all_boards = generate_all_legal_boards()
    print(f"Gefunden: {len(all_boards)} legale Boards")

    if filter_symmetries:
        # Optional: reduzierte Variante über Symmetrieklassen
        print("Filtere Symmetrien...")
        canonical_boards = [board for board in all_boards if is_canonical(board)]
        print(f"Verbleibend nach Symmetrie-Filter: {len(canonical_boards)} Boards")
    else:
        # Standard: alle legalen Boards behalten (bessere Generalisierung im echten Spiel)
        canonical_boards = all_boards
        print(f"Symmetrie-Filter deaktiviert: {len(canonical_boards)} Boards")
    
    # Sammle Daten mit optimalen Moves
    data = []
    phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}
    
    # Minimax-Cache für schnellere Berechnung
    minimax_cache = {}
    
    print(f"Berechne optimale Moves (multi_hot={multi_hot})...")
    for board in tqdm(canonical_boards):
        player = get_current_player(board)
        empty_count = np.sum(board == 0)
        
        # Bestimme Phase
        if empty_count >= 6:
            phase = 'opening'
        elif empty_count >= 3:
            phase = 'midgame'
        else:
            phase = 'endgame'
        
        # Berechne optimale Moves mit taktischer Priorisierung
        optimal_moves = find_tactical_best_moves_for_player(board, player)
        
        if not optimal_moves:
            continue  # Keine legalen Moves
        
        # Kanonisierung
        canonical_board = canonicalize_board(board, player)
        
        if multi_hot:
            # Multi-Hot Format: 9-dim vektor, 1 für alle optimalen Moves
            label = np.zeros(9, dtype=int)
            for move in optimal_moves:
                label[move] = 1
            data.append(np.concatenate([canonical_board, label]))
        else:
            # Kompatibilität: Nur erster optimaler Move
            label = np.zeros(9, dtype=int)
            label[optimal_moves[0]] = 1
            data.append(np.concatenate([canonical_board, [label.argmax()]]))
        
        phase_counts[phase] += 1
        
        # Balance: Stop wenn genug Samples pro Phase
        if balance_phases and len(data) >= target_samples:
            break
    
    print(f"\nPhase-Verteilung:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} Boards")
    
    result = np.array(data, dtype=int)
    print(f"\nFinaler Datensatz: {len(result)} Samples")
    print(f"Shape: {result.shape}")
    
    return result


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
    parser.add_argument(
        "-m", "--mode",
        choices=["tree", "selfplay", "optimal"],
        default="tree",
        help="Generierungsmodus: 'tree' (alles wie bisher), 'selfplay' (realistische Partien), 'optimal' (systematisch, ausgewogen)"
    )
    args = parser.parse_args()
    
    if args.mode == "optimal":
        print("Verwende NEUEN optimalen Generator...")
        dataset = generate_optimal_dataset(target_samples=args.size, balance_phases=True)
    else:
        dataset = generate_dataset(max_boards=args.size, mode=args.mode)
    
    num_samples = len(dataset)
    print("Datensatz fertig. Anzahl Boards:", num_samples)
    
    # In datasets-Ordner speichern
    datasets_dir = "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    filename = f"tictactoe_dataset_{num_samples}_{args.mode}.npy"
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


