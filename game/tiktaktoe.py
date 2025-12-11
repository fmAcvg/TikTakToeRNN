import numpy as np

class TicTacToe:
    def __init__(self):
        # Spielfeld initialisieren ← NEU
        self.board = np.zeros((3, 3), dtype=object)  # ← GEÄNDERT (NumPy statt Liste)
        self.current_player = "1"  # ← NEU

    def print_board(self):
        print()
        for row in self.board:
            print(" | ".join(str(x) for x in row))
        print()

    def check_win(self):
        # Reihen prüfen
        for row in self.board:
            if isinstance(row[0], str) and np.all(row == row[0]):  # ← GEÄNDERT (NumPy Vergleich)
                return "win"  # ← GEÄNDERT

        # Spalten prüfen
        for col in range(3):
            column = self.board[:, col]  # ← GEÄNDERT (NumPy Spaltenzugriff)
            if isinstance(column[0], str) and np.all(column == column[0]):  # ← GEÄNDERT
                return "win"  # ← GEÄNDERT

        # Diagonalen prüfen
        diag1 = np.diag(self.board)  # ← GEÄNDERT (NumPy diag)
        if isinstance(diag1[0], str) and np.all(diag1 == diag1[0]):  # ← GEÄNDERT
            return "win"  # ← GEÄNDERT

        diag2 = np.diag(np.fliplr(self.board))  # ← GEÄNDERT (NumPy umgedrehte Diagonale)
        if isinstance(diag2[0], str) and np.all(diag2 == diag2[0]):  # ← GEÄNDERT
            return "win"  # ← GEÄNDERT

        # Unentschieden prüfen
        full = not np.any(self.board == 0)  # ← GEÄNDERT (NumPy any)
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
            if self.board[row, col] in ["1", "-1"]:
                print("Dieses Feld ist schon belegt!")
                continue

            # setzen
            self.board[row, col] = self.current_player  # ← GEÄNDERT

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
            self.current_player = "-1" if self.current_player == "1" else "1"  # ← GEÄNDERT


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