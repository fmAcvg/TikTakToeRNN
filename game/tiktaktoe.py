import numpy as np

class TicTacToe:
    def __init__(self):
        # Spielfeld initialisieren ← NEU
        self.board = [[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]]
        self.current_player = "x"  # ← NEU

    def print_board(self):
        print()
        for row in self.board:
            print(" | ".join(str(x) for x in row))
        print()

    def check_win(self):
        # Reihen prüfen
        for row in self.board:
            if row[0] == row[1] == row[2] and isinstance(row[0], str):  # ← GEÄNDERT
                return "win"  # ← GEÄNDERT

        # Spalten prüfen
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and isinstance(self.board[0][col], str):  # ← GEÄNDERT
                return "win"  # ← GEÄNDERT

        # Diagonalen prüfen
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and isinstance(self.board[0][0], str):  # ← GEÄNDERT
            return "win"  # ← GEÄNDERT

        if self.board[0][2] == self.board[1][1] == self.board[2][0] and isinstance(self.board[0][2], str):  # ← GEÄNDERT
            return "win"  # ← GEÄNDERT

        # Unentschieden prüfen
        full = all(not isinstance(val, int) for row in self.board for val in row)  # ← GEÄNDERT
        if full:
            return "draw"  # ← NEU

        return False  # ← GEÄNDERT

    def play(self):
        while True:
            self.print_board()
            print(f"Spieler {self.current_player} ist dran.")

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
            if self.board[row][col] in ["x", "o"]:
                print("Dieses Feld ist schon belegt!")
                continue

            # setzen
            self.board[row][col] = self.current_player  # ← GEÄNDERT

            # prüfen Sieg / Unentschieden
            result = self.check_win()  # ← GEÄNDERT

            if result == "win":  # ← GEÄNDERT
                self.print_board()
                print(f"Spieler {self.current_player} hat gewonnen!")  # ← GEÄNDERT
                break
            if result == "draw":  # ← GEÄNDERT
                self.print_board()
                print("Unentschieden!")  # ← GEÄNDERT
                break

            # Spieler wechseln
            self.current_player = "o" if self.current_player == "x" else "x"  # ← GEÄNDERT


# Start ← NEU
game = TicTacToe()  # ← NEU
game.play()  # ← NEU


#bei markierung durch geändert und durch neu, Quelle durch chatgpt:

'''Ich habe ein Tic-Tac-Toe-Spiel in Python. Im aktuellen Code erkennt die Funktion check_win() nicht, 
ob jemand gewonnen hat oder ob das Spiel unentschieden endet. 
Außerdem wird beim Unentschieden fälschlicherweise angezeigt, dass ein Spieler gewonnen hat. 
Bitte passe den Code so an, dass:

check_win() "win" zurückgibt, wenn ein Spieler gewinnt, "draw" wenn das Spiel unentschieden endet und False, 
wenn das Spiel weitergeht.

Die play()-Funktion die Rückgabe korrekt auswertet, Gewinner korrekt anzeigt und Unentschieden korrekt ausgibt.

Spielerwechsel nur bei gültigem Zug erfolgt'''