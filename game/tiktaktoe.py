import numpy as np

field = [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]


def print_field(board):
    print()
    for row in board:
        print(" | ".join(str(x) for x in row))
    print()


def check_win(board):
    # Reihen prüfen
    for row in board:
        if row[0] == row[1] == row[2] and isinstance(row[0], str):  # ← GEÄNDERT
            return "win"  # ← GEÄNDERT

    # Spalten prüfen
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and isinstance(board[0][col], str):  # ← GEÄNDERT
            return "win"

    # Diagonalen prüfen
    if board[0][0] == board[1][1] == board[2][2] and isinstance(board[0][0], str):  # ← GEÄNDERT
        return "win"

    if board[0][2] == board[1][1] == board[2][0] and isinstance(board[0][2], str):  # ← GEÄNDERT
        return "win"

    # Unentschieden prüfen
    full = True
    for row in board:
        for val in row:
            if isinstance(val, int):  # ← GEÄNDERT
                full = False

    if full:
        return "draw"  # ← NEU

    return False


def play(board):
    current = "x"

    while True:
        print_field(board)
        print(f"Spieler {current} ist dran.")

        # Eingabe
        try:
            choice = int(input("Wähle ein Feld (0-8): "))
        except ValueError:
            print("Bitte eine Zahl von 0 bis 8 eingeben!")
            continue

        if choice < 0 or choice > 8:
            print("Ungültige Zahl! Wähle 0 bis 8.")
            continue

        row = choice // 3
        col = choice % 3

        # prüfen ob das Feld frei ist
        if board[row][col] in ["x", "o"]:
            print("Dieses Feld ist schon belegt!")
            continue

        # setzen
        board[row][col] = current

        # prüfen Sieg / Unentschieden
        result = check_win(board)  # ← GEÄNDERT

        if result == "win":  # ← GEÄNDERT
            print_field(board)
            print(f"Spieler {current} hat gewonnen!")  # ← GEÄNDERT
            break
        if result == "draw":  # ← GEÄNDERT
            print_field(board)
            print("Unentschieden!")  # ← GEÄNDERT
            break

        # Spieler wechseln
        current = "o" if current == "x" else "x"


# Start
play(field)



#bei markierung durch geändert und durch neu, Quelle durch chatgpt:

'''Ich habe ein Tic-Tac-Toe-Spiel in Python. Im aktuellen Code erkennt die Funktion check_win() nicht, 
ob jemand gewonnen hat oder ob das Spiel unentschieden endet. 
Außerdem wird beim Unentschieden fälschlicherweise angezeigt, dass ein Spieler gewonnen hat. 
Bitte passe den Code so an, dass:

check_win() "win" zurückgibt, wenn ein Spieler gewinnt, "draw" wenn das Spiel unentschieden endet und False, 
wenn das Spiel weitergeht.

Die play()-Funktion die Rückgabe korrekt auswertet, Gewinner korrekt anzeigt und Unentschieden korrekt ausgibt.

Spielerwechsel nur bei gültigem Zug erfolgt'''