#lenny
import tkinter as tk
from tiktaktoe import TicTacToe  # TicTacToe klasse importieren


# TicTacToe objekt erstellen

game = TicTacToe()


root = tk.Tk()
root.title("TicTacToe")
root.geometry("500x500")


# Label für spieleranzeigee

label_player = tk.Label(root, text=f'Spieler {game.current_player} ist dran')
label_player.grid(row=0, column=0, columnspan=3)  # label oben über grid


# 3x3 buttons vorbereiten

buttons = [[None for _ in range(3)] for _ in range(3)]


# funktion, die beim click auf einen button ausgeführt wird

def click(row, col):
    # prüfen ob  Feld schon belegt ist
    if game.board[row, col] in ['1', '-1']:
        return

    # spielerzeichen ins Board setzen
    game.board[row, col] = game.current_player
    buttons[row][col].config(text=game.current_player)  # Buttontext aktualisieren

    # Sieg oder Unentschieden prüfen
    result = game.check_win()
    if result == 'win':
        label_player.config(text=f'Spieler {game.current_player} hat gewonnen!')
        #alle Buttons deaktivieren
        for r in range(3):
            for c in range(3):
                buttons[r][c].config(state="disabled")
        return
    elif result == "draw":
        label_player.config(text="Unentschieden!")
        return

    # Spieler wechseln
    game.current_player = "-1" if game.current_player == "1" else "1"
    label_player.config(text=f"Spieler {game.current_player} ist dran")


# Buttons erstellen und in grid legen

for r in range(3):
    for c in range(3):
        btn = tk.Button(
            root, text="", width=10, height=5,
            command=lambda r=r, c=c: click(r, c)  # lambda speichert aktuelle Koordinaten
        )
        btn.grid(row=r + 1, column=c)  # +1 weil row 0 für Label
        buttons[r][c] = btn  # Button in liste speichern


root.mainloop()



