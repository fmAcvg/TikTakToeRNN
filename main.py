import numpy as np
import os
import glob
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from NeuralNetwork.predict import NeuralNetwork, create_layer
from NeuralNetwork.training import train_on_data_set
from NeuralNetwork.rnn.neuron import Neuron
from NeuralNetwork.rnn.Layer import Layer

def get_available_datasets():
    """Gibt eine Liste aller verfügbaren Datasets zurück"""
    dataset_dir = "datasets"
    
    # Suche nach allen Datasets mit Pattern tictactoe_dataset_*.npy
    pattern = os.path.join(dataset_dir, "tictactoe_dataset_*.npy")
    datasets = glob.glob(pattern)
    
    # Fallback: altes Format ohne Anzahl
    old_path = os.path.join(dataset_dir, "tictactoe_dataset.npy")
    if os.path.exists(old_path):
        datasets.append(old_path)
    
    # Sortiere nach Modifikationszeit (neueste zuerst)
    datasets.sort(key=os.path.getmtime, reverse=True)
    
    # Extrahiere nur die Dateinamen für die Anzeige
    dataset_names = [os.path.basename(d) for d in datasets]
    return datasets, dataset_names

def rotate_board_90(board, move):
    """Rotiert Board um 90° im Uhrzeigersinn und passt den Zug entsprechend an"""
    b = board.reshape(3, 3)
    rotated_b = np.rot90(b, k=-1)  # -1 = 90° im Uhrzeigersinn
    rotated_board = rotated_b.flatten()
    
    # Transformiere den Zug: Rotation-Mapping
    # 0->2, 1->5, 2->8, 3->1, 4->4, 5->7, 6->0, 7->3, 8->6
    rotation_map = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    rotated_move = rotation_map[move]
    
    return rotated_board, rotated_move

def mirror_board_horizontal(board, move):
    """Spiegelt Board horizontal und passt den Zug entsprechend an"""
    b = board.reshape(3, 3)
    mirrored_b = np.fliplr(b)
    mirrored_board = mirrored_b.flatten()
    
    # Transformiere den Zug: Horizontal-Spiegelung-Mapping
    # 0->2, 1->1, 2->0, 3->5, 4->4, 5->3, 6->8, 7->7, 8->6
    mirror_map = [2, 1, 0, 5, 4, 3, 8, 7, 6]
    mirrored_move = mirror_map[move]
    
    return mirrored_board, mirrored_move

def augment_board(board, move):
    """Generiert alle 8 Varianten eines Boards (4 Rotationen × 2 Spiegelungen)"""
    variants = []
    
    # Original
    variants.append((board.copy(), move))
    
    # 3 Rotationen (90°, 180°, 270°)
    current_board, current_move = board.copy(), move
    for _ in range(3):
        current_board, current_move = rotate_board_90(current_board, current_move)
        variants.append((current_board.copy(), current_move))
    
    # Horizontal gespiegelt + 4 Rotationen
    mirrored_board, mirrored_move = mirror_board_horizontal(board, move)
    variants.append((mirrored_board.copy(), mirrored_move))
    
    current_board, current_move = mirrored_board.copy(), mirrored_move
    for _ in range(3):
        current_board, current_move = rotate_board_90(current_board, current_move)
        variants.append((current_board.copy(), current_move))
    
    return variants

def load_dataset(dataset_path=None, use_augmentation=False):
    """Lädt das Dataset aus training-dataset
    
    Args:
        dataset_path: Pfad zum Dataset (None = neuestes Dataset)
        use_augmentation: Wenn True, werden alle Boards augmentiert (8x mehr Daten)
    """
    if dataset_path is None:
        # Fallback: neuestes Dataset
        datasets, _ = get_available_datasets()
        if not datasets:
            return None
        dataset_path = datasets[0]
    
    if not os.path.exists(dataset_path):
        return None
    
    data = np.load(dataset_path)
    # Daten aufteilen: erste 9 Werte = Board, letzter Wert = bester Zug
    boards = data[:, :9].astype(float)
    moves = data[:, 9].astype(int)
    
    # Data Augmentation
    if use_augmentation:
        augmented_boards = []
        augmented_moves = []
        for i in range(len(boards)):
            variants = augment_board(boards[i], moves[i])
            for variant_board, variant_move in variants:
                augmented_boards.append(variant_board)
                augmented_moves.append(variant_move)
        boards = np.array(augmented_boards)
        moves = np.array(augmented_moves)
    
    # One-Hot-Encoding für die Züge
    one_hot_moves = np.zeros((len(moves), 9))
    one_hot_moves[np.arange(len(moves)), moves] = 1
    
    return boards, one_hot_moves

def get_available_models():
    """Gibt eine Liste aller verfügbaren Modelle zurück (nur aus dem models-Ordner)"""
    models_dir = "models"
    models_pattern = os.path.join(models_dir, "model_*.npz")
    
    all_models = glob.glob(models_pattern)
    
    # Sortiere nach Modifikationszeit (neueste zuerst)
    all_models.sort(key=os.path.getmtime, reverse=True)
    
    # Extrahiere nur die Dateinamen für die Anzeige
    model_names = [os.path.basename(m) for m in all_models]
    return all_models, model_names

def load_model(model_path, num_hidden_layers, hidden_neurons):
    """Lädt ein gespeichertes Modell aus einer .npz Datei"""
    if not os.path.exists(model_path):
        return None
    
    data = np.load(model_path)
    
    # Netzwerk-Struktur erstellen
    input_size = 9
    output_size = 9
    
    layers = []
    
    # Erster Hidden-Layer
    if num_hidden_layers > 0:
        layer0_neurons = []
        for j in range(hidden_neurons):
            weights = data[f'layer_0_neuron_{j}_weights']
            bias = data[f'layer_0_neuron_{j}_bias']
            layer0_neurons.append(Neuron(weights, bias))
        layers.append(Layer(layer0_neurons))
        current_size = hidden_neurons
        
        # Weitere Hidden-Layer
        for i in range(1, num_hidden_layers):
            layer_neurons = []
            for j in range(hidden_neurons):
                weights = data[f'layer_{i}_neuron_{j}_weights']
                bias = data[f'layer_{i}_neuron_{j}_bias']
                layer_neurons.append(Neuron(weights, bias))
            layers.append(Layer(layer_neurons))
    else:
        current_size = input_size
    
    # Output-Layer
    output_layer_idx = num_hidden_layers
    output_neurons = []
    for j in range(output_size):
        weights = data[f'layer_{output_layer_idx}_neuron_{j}_weights']
        bias = data[f'layer_{output_layer_idx}_neuron_{j}_bias']
        output_neurons.append(Neuron(weights, bias))
    layers.append(Layer(output_neurons))
    
    model = NeuralNetwork(layers)
    return model

def generate_random_board():
    """Generiert einen zufälligen gültigen Tic-Tac-Toe Spielstand"""
    board = np.zeros(9, dtype=int)
    
    # Zufällige Anzahl von Zügen (1-8, da mindestens ein Feld frei bleiben muss)
    num_moves = np.random.randint(1, 9)
    
    # Zufällige Reihenfolge der Felder
    positions = np.random.permutation(9)
    
    # Fülle das Board abwechselnd mit 1 und -1
    for i in range(num_moves):
        pos = positions[i]
        player = 1 if i % 2 == 0 else -1
        board[pos] = player
    
    return board

def is_valid_board(board):
    """Prüft ob ein Board gültig ist (nicht gewonnen, nicht voll)"""
    # Prüfe auf Gewinner
    b = board.reshape(3, 3)
    lines = [
        b[0], b[1], b[2],
        b[:, 0], b[:, 1], b[:, 2],
        b.diagonal(), np.fliplr(b).diagonal()
    ]
    
    for line in lines:
        if np.all(line == 1) or np.all(line == -1):
            return False
    
    # Prüfe ob voll
    if np.all(board != 0):
        return False
    
    return True

def generate_test_boards(num_boards=100):
    """Generiert zufällige gültige Test-Boards"""
    boards = []
    attempts = 0
    max_attempts = num_boards * 10
    
    while len(boards) < num_boards and attempts < max_attempts:
        board = generate_random_board()
        if is_valid_board(board):
            boards.append(board)
        attempts += 1
    
    return np.array(boards)

def test_model_on_random_boards(model, num_test_boards=100):
    """Testet ein Modell auf zufälligen Spielständen"""
    test_boards = generate_test_boards(num_test_boards)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    find_best_move_for_player = dataset_module.find_best_move_for_player
    get_current_player = dataset_module.get_current_player
    
    correct_predictions = 0
    total_tests = 0
    
    for board in test_boards:
        if not is_valid_board(board):
            continue
        
        player = get_current_player(board)
        optimal_move = find_best_move_for_player(board, player)
        if optimal_move is None:
            continue
        
        # Modell-Vorhersage
        predicted_move, _ = model.predict(board.astype(float))
        
        if predicted_move == optimal_move:
            correct_predictions += 1
        total_tests += 1
    
    accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    return accuracy, total_tests, correct_predictions

def test_on_realistic_games(model, num_test_boards=100):
    """Testet ein Modell auf realistischen Spielverläufen (wie im Training)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    find_best_move_for_player = dataset_module.find_best_move_for_player
    get_next_boards = dataset_module.get_next_boards
    check_winner = dataset_module.check_winner
    
    # Generiere realistische Spielverläufe (BFS) - speichere (board, player)
    test_states = []
    start_board = np.zeros(9, dtype=int)
    states = [(start_board, 1)]
    
    while states and len(test_states) < num_test_boards:
        board, player = states.pop(0)
        winner = check_winner(board)
        if winner != 0 or np.all(board != 0):
            continue
        
        test_states.append((board.copy(), player))
        for next_board in get_next_boards(board, player):
            states.append((next_board, -player))
    
    correct_predictions = 0
    total_tests = 0
    
    for board, player in test_states:
        optimal_move = find_best_move_for_player(board, player)
        if optimal_move is None:
            continue
        
        # Modell-Vorhersage
        predicted_move, _ = model.predict(board.astype(float))
        
        if predicted_move == optimal_move:
            correct_predictions += 1
        total_tests += 1
    
    accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    return accuracy, total_tests, correct_predictions

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TikTakToe RNN Training")
        self.root.geometry("800x600")
        
        self.is_training = False
        self.stop_training = False
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.epoch_history = []
        self.current_model = None
        
        # Game state für Play Tab
        self.game_board = np.zeros(9, dtype=int)
        self.game_model = None
        self.game_active = False
        self.player_turn = True  # True = Spieler (X), False = Modell (O)
        self.player_symbol = 1  # Spieler ist X (1)
        self.model_symbol = -1  # Modell ist O (-1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Notebook für Tabs erstellen
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        
        # Tab 2: Spielen
        self.play_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.play_tab, text="Spielen")
        
        # Training Tab Setup
        self.setup_training_tab()
        
        # Play Tab Setup
        self.setup_play_tab()
    
    def setup_training_tab(self):
        # Eingabefelder
        input_frame = ttk.Frame(self.training_tab, padding="10")
        input_frame.pack(fill=tk.X)
        
        # Erste Zeile: Epochen, Lernrate, Button
        ttk.Label(input_frame, text="Epochen:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="150")
        ttk.Entry(input_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Lernrate:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.01")
        ttk.Entry(input_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        self.start_button = ttk.Button(input_frame, text="Training starten", command=self.start_training)
        self.start_button.grid(row=0, column=4, padx=5, pady=5)
        self.stop_button = ttk.Button(input_frame, text="Training stoppen", command=self.stop_training_request, state="disabled")
        self.stop_button.grid(row=0, column=5, padx=5, pady=5)
        
        # Zweite Zeile: Hidden-Layer Anzahl, Neuronen pro Layer
        ttk.Label(input_frame, text="Anzahl Hidden-Layer:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_layers_var = tk.StringVar(value="3")
        ttk.Entry(input_frame, textvariable=self.num_layers_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Neuronen pro Hidden-Layer:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.hidden_neurons_var = tk.StringVar(value="32")
        ttk.Entry(input_frame, textvariable=self.hidden_neurons_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Dritte Zeile: Dataset auswählen
        ttk.Label(input_frame, text="Dataset:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.dataset_paths, self.dataset_names = get_available_datasets()
        self.dataset_var = tk.StringVar()
        if self.dataset_names:
            self.dataset_var.set(self.dataset_names[0])
        self.dataset_combo = ttk.Combobox(input_frame, textvariable=self.dataset_var, values=self.dataset_names, 
                                         state="readonly", width=30)
        self.dataset_combo.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        refresh_button = ttk.Button(input_frame, text="Aktualisieren", command=self.refresh_datasets)
        refresh_button.grid(row=2, column=3, padx=5, pady=5)
        
        # Vierte Zeile: Modell auswählen und testen
        ttk.Label(input_frame, text="Modell:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_paths, self.model_names = get_available_models()
        self.model_var = tk.StringVar()
        if self.model_names:
            self.model_var.set(self.model_names[0])
        self.model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, values=self.model_names, 
                                       state="readonly", width=30)
        self.model_combo.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.test_button = ttk.Button(input_frame, text="Modell testen", command=self.test_model)
        self.test_button.grid(row=3, column=3, padx=5, pady=5)
        self.save_current_button = ttk.Button(input_frame, text="Aktuelles Modell speichern", command=self.save_current_model)
        self.save_current_button.grid(row=3, column=4, padx=5, pady=5)
        
        # Status
        self.status_label = ttk.Label(input_frame, text="Bereit", foreground="green")
        self.status_label.grid(row=4, column=0, columnspan=5, pady=5)
        
        # Chart
        self.setup_chart()
        
    def setup_chart(self):
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_loss = self.fig.add_subplot(111)
        self.ax_loss.set_xlabel("Epoche")
        self.ax_loss.set_ylabel("Loss", color='b')
        self.ax_loss.tick_params(axis='y', labelcolor='b')
        self.ax_loss.grid(True)
        
        # Zweite y-Achse für Accuracy
        self.ax_acc = self.ax_loss.twinx()
        self.ax_acc.set_ylabel("Accuracy (%)", color='r')
        self.ax_acc.tick_params(axis='y', labelcolor='r')
        
        self.fig.suptitle("Training Progress: Loss & Accuracy", fontsize=12)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.training_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def update_chart(self):
        if len(self.epoch_history) > 0:
            self.ax_loss.clear()
            self.ax_acc.clear()
            
            # Loss plotten (Train und Test)
            self.ax_loss.plot(self.epoch_history, self.loss_history, 'b-', label='Train Loss', linewidth=2)
            if len(self.test_loss_history) > 0:
                self.ax_loss.plot(self.epoch_history, self.test_loss_history, 'b--', label='Test Loss', linewidth=2)
            self.ax_loss.set_xlabel("Epoche")
            self.ax_loss.set_ylabel("Loss", color='b')
            self.ax_loss.tick_params(axis='y', labelcolor='b')
            self.ax_loss.grid(True)
            
            # Train + Test Accuracy plotten (Overfitting sichtbar)
            if len(self.train_accuracy_history) > 0:
                self.ax_acc.plot(self.epoch_history, self.train_accuracy_history, 'g-', label='Train Acc', linewidth=2)
            if len(self.test_accuracy_history) > 0:
                self.ax_acc.plot(self.epoch_history, self.test_accuracy_history, 'r-', label='Test Acc', linewidth=2)
            self.ax_acc.set_ylabel("Accuracy (%)", color='r')
            self.ax_acc.tick_params(axis='y', labelcolor='r')
            
            lines1, labels1 = self.ax_loss.get_legend_handles_labels()
            lines2, labels2 = self.ax_acc.get_legend_handles_labels()
            self.ax_loss.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            self.fig.suptitle("Train/Test Loss & Train/Test Accuracy (Overfitting)", fontsize=12)
            self.canvas.draw()
    
    def setup_play_tab(self):
        """Erstellt die UI für das Spielen gegen das Modell"""
        # Modell-Auswahl
        model_frame = ttk.Frame(self.play_tab, padding="10")
        model_frame.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Modell zum Spielen:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.play_model_paths, self.play_model_names = get_available_models()
        self.play_model_var = tk.StringVar()
        if self.play_model_names:
            self.play_model_var.set(self.play_model_names[0])
        self.play_model_combo = ttk.Combobox(model_frame, textvariable=self.play_model_var, 
                                            values=self.play_model_names, state="readonly", width=30)
        self.play_model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(model_frame, text="Hidden-Layer:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.play_num_layers_var = tk.StringVar(value="3")
        ttk.Entry(model_frame, textvariable=self.play_num_layers_var, width=5).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Neuronen:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.play_hidden_neurons_var = tk.StringVar(value="32")
        ttk.Entry(model_frame, textvariable=self.play_hidden_neurons_var, width=5).grid(row=0, column=5, padx=5, pady=5)
        
        self.load_model_button = ttk.Button(model_frame, text="Modell laden", command=self.load_game_model)
        self.load_model_button.grid(row=0, column=6, padx=5, pady=5)
        
        self.new_game_button = ttk.Button(model_frame, text="Neues Spiel", command=self.start_new_game)
        self.new_game_button.grid(row=0, column=7, padx=5, pady=5)
        
        # Status
        self.game_status_label = ttk.Label(model_frame, text="Lade ein Modell und starte ein neues Spiel", foreground="blue")
        self.game_status_label.grid(row=1, column=0, columnspan=8, pady=10)
        
        # Spielbrett
        board_frame = ttk.Frame(self.play_tab, padding="20")
        board_frame.pack(expand=True)
        
        self.board_buttons = []
        for i in range(9):
            row = i // 3
            col = i % 3
            btn = tk.Button(board_frame, text="", font=("Arial", 24, "bold"), 
                           width=4, height=2, command=lambda idx=i: self.make_move(idx))
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.board_buttons.append(btn)
    
    def load_game_model(self):
        """Lädt das ausgewählte Modell für das Spiel"""
        try:
            num_hidden_layers = int(self.play_num_layers_var.get())
            hidden_neurons = int(self.play_hidden_neurons_var.get())
        except ValueError:
            self.game_status_label.config(text="Fehler: Ungültige Netzwerk-Parameter", foreground="red")
            return
        
        selected_name = self.play_model_var.get()
        if not selected_name:
            self.game_status_label.config(text="Kein Modell ausgewählt", foreground="red")
            return
        
        model_path = None
        if selected_name and self.play_model_paths:
            try:
                idx = self.play_model_names.index(selected_name)
                model_path = self.play_model_paths[idx]
            except (ValueError, IndexError):
                self.game_status_label.config(text="Fehler: Modell nicht gefunden", foreground="red")
                return
        
        try:
            self.game_model = load_model(model_path, num_hidden_layers, hidden_neurons)
            if self.game_model is None:
                self.game_status_label.config(text="Fehler: Modell konnte nicht geladen werden", foreground="red")
                return
            
            self.game_status_label.config(text="Modell geladen! Klicke auf 'Neues Spiel' um zu beginnen.", foreground="green")
        except Exception as e:
            self.game_status_label.config(text=f"Fehler beim Laden: {str(e)}", foreground="red")
    
    def start_new_game(self):
        """Startet ein neues Spiel"""
        if self.game_model is None:
            self.game_status_label.config(text="Bitte lade zuerst ein Modell!", foreground="red")
            return
        
        # Board zurücksetzen
        self.game_board = np.zeros(9, dtype=int)
        self.game_active = True
        self.player_turn = True  # Spieler beginnt
        
        # Buttons zurücksetzen
        for btn in self.board_buttons:
            btn.config(text="", state=tk.NORMAL, bg="SystemButtonFace")
        
        self.game_status_label.config(text="Dein Zug (X). Klicke auf ein Feld.", foreground="blue")
    
    def make_move(self, position):
        """Spieler macht einen Zug"""
        if not self.game_active or not self.player_turn:
            return
        
        if self.game_board[position] != 0:
            return  # Feld bereits belegt
        
        # Spieler-Zug
        self.game_board[position] = self.player_symbol
        self.board_buttons[position].config(text="X", state=tk.DISABLED, bg="lightblue")
        
        # Prüfe auf Gewinner oder Unentschieden
        winner = self.check_winner()
        if winner != 0:
            self.end_game(winner)
            return
        
        if np.all(self.game_board != 0):
            self.end_game(0)  # Unentschieden
            return
        
        # Modell ist dran
        self.player_turn = False
        self.game_status_label.config(text="Modell denkt nach...", foreground="orange")
        self.root.after(100, self.model_move)  # Kurze Verzögerung für bessere UX
    
    def model_move(self):
        """Modell macht einen Zug"""
        if not self.game_active or self.player_turn:
            return
        
        # Modell-Vorhersage
        try:
            predicted_move, probs = self.game_model.predict(self.game_board.astype(float))
            
            # Finde das beste freie Feld
            valid_moves = [i for i in range(9) if self.game_board[i] == 0]
            if not valid_moves:
                self.end_game(0)  # Unentschieden
                return
            
            # Wenn vorhergesagter Zug ungültig ist, nimm das beste freie Feld
            if predicted_move not in valid_moves:
                # Sortiere nach Wahrscheinlichkeit und nimm das beste freie Feld
                move_probs = [(i, probs[i]) for i in valid_moves]
                move_probs.sort(key=lambda x: x[1], reverse=True)
                predicted_move = move_probs[0][0]
            
            # Modell-Zug
            self.game_board[predicted_move] = self.model_symbol
            self.board_buttons[predicted_move].config(text="O", state=tk.DISABLED, bg="lightcoral")
            
            # Prüfe auf Gewinner oder Unentschieden
            winner = self.check_winner()
            if winner != 0:
                self.end_game(winner)
                return
            
            if np.all(self.game_board != 0):
                self.end_game(0)  # Unentschieden
                return
            
            # Spieler ist wieder dran
            self.player_turn = True
            self.game_status_label.config(text="Dein Zug (X). Klicke auf ein Feld.", foreground="blue")
            
        except Exception as e:
            self.game_status_label.config(text=f"Fehler beim Modell-Zug: {str(e)}", foreground="red")
            self.game_active = False
    
    def check_winner(self):
        """Prüft ob es einen Gewinner gibt. Gibt 1 (Spieler), -1 (Modell) oder 0 (kein Gewinner) zurück"""
        b = self.game_board.reshape(3, 3)
        lines = [
            b[0], b[1], b[2],  # rows
            b[:, 0], b[:, 1], b[:, 2],  # columns
            b.diagonal(), np.fliplr(b).diagonal()  # diagonals
        ]
        
        for line in lines:
            if np.all(line == 1):
                return 1  # Spieler gewinnt
            if np.all(line == -1):
                return -1  # Modell gewinnt
        
        return 0  # Kein Gewinner
    
    def end_game(self, winner):
        """Beendet das Spiel und zeigt das Ergebnis"""
        self.game_active = False
        
        # Alle Buttons deaktivieren
        for btn in self.board_buttons:
            btn.config(state=tk.DISABLED)
        
        if winner == 1:
            self.game_status_label.config(text="🎉 Du hast gewonnen!", foreground="green")
            # Gewinnende Linie hervorheben
            self.highlight_winning_line()
        elif winner == -1:
            self.game_status_label.config(text="🤖 Modell hat gewonnen!", foreground="red")
            self.highlight_winning_line()
        else:
            self.game_status_label.config(text="🤝 Unentschieden!", foreground="orange")
    
    def highlight_winning_line(self):
        """Hebt die gewinnende Linie hervor"""
        b = self.game_board.reshape(3, 3)
        lines = [
            (b[0], [0, 1, 2]), (b[1], [3, 4, 5]), (b[2], [6, 7, 8]),  # rows
            (b[:, 0], [0, 3, 6]), (b[:, 1], [1, 4, 7]), (b[:, 2], [2, 5, 8]),  # columns
            (b.diagonal(), [0, 4, 8]), (np.fliplr(b).diagonal(), [2, 4, 6])  # diagonals
        ]
        
        for line, indices in lines:
            if np.all(line == 1) or np.all(line == -1):
                for idx in indices:
                    self.board_buttons[idx].config(bg="yellow")
                break
    
    def start_training(self):
        if self.is_training:
            return
        
        try:
            epochs = int(self.epochs_var.get())
            learning_rate = float(self.lr_var.get())
            num_hidden_layers = int(self.num_layers_var.get())
            hidden_neurons = int(self.hidden_neurons_var.get())
            
            if num_hidden_layers < 1:
                raise ValueError("Anzahl Hidden-Layer muss >= 1 sein")
            if hidden_neurons < 1:
                raise ValueError("Anzahl Neuronen muss >= 1 sein")
        except ValueError as e:
            self.status_label.config(text=f"Fehler: {str(e)}", foreground="red")
            return
        
        self.is_training = True
        self.stop_training = False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.epoch_history = []
        
        # Training in separatem Thread starten
        thread = threading.Thread(target=self.train_model, args=(epochs, learning_rate, num_hidden_layers, hidden_neurons))
        thread.daemon = True
        thread.start()
    
    def stop_training_request(self):
        """Setzt das Stopp-Flag, damit die Trainingsschleife abbricht"""
        self.stop_training = True
    
    def train_model(self, epochs, learning_rate, num_hidden_layers, hidden_neurons):
        try:
            self.root.after(0, self.status_label.config, {"text": "Lade Dataset...", "foreground": "blue"})
            
            # Lade das ausgewählte Dataset
            selected_name = self.dataset_var.get()
            dataset_path = None
            if selected_name and self.dataset_paths:
                try:
                    idx = self.dataset_names.index(selected_name)
                    dataset_path = self.dataset_paths[idx]
                except (ValueError, IndexError):
                    pass
            
            boards, correct_moves = load_dataset(dataset_path, use_augmentation=False)
            if boards is None:
                self.root.after(0, self.status_label.config, {"text": "Fehler: Dataset nicht gefunden", "foreground": "red"})
                return
            
            # Train/Test Split (80/20)
            split_idx = int(len(boards) * 0.8)
            indices = np.random.permutation(len(boards))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            train_boards = boards[train_indices]
            train_moves = correct_moves[train_indices]
            test_boards = boards[test_indices]
            test_moves = correct_moves[test_indices]
            
            self.root.after(0, self.status_label.config, {"text": f"Dataset geladen: {len(train_boards)} Train, {len(test_boards)} Test", "foreground": "blue"})
            
            # Netzwerk dynamisch erstellen
            input_size = 9
            output_size = 9
            
            layers = []
            
            # Erster Hidden-Layer (von Input)
            if num_hidden_layers > 0:
                layers.append(create_layer(input_size, hidden_neurons))
                current_size = hidden_neurons
                
                # Weitere Hidden-Layer
                for _ in range(num_hidden_layers - 1):
                    layers.append(create_layer(current_size, hidden_neurons))
                    current_size = hidden_neurons
            else:
                current_size = input_size
            
            # Output-Layer
            layers.append(create_layer(current_size, output_size))
            
            model = NeuralNetwork(layers)
            
            self.root.after(0, self.status_label.config, {"text": f"Netzwerk erstellt: {num_hidden_layers} Hidden-Layer mit je {hidden_neurons} Neuronen", "foreground": "blue"})
            
            self.status_label.config(text=f"Training läuft... (0/{epochs})", foreground="blue")
            
            # Learning Rate Scheduling: Reduziere LR über Zeit
            initial_lr = learning_rate
            
            # Training
            for epoch in range(epochs):
                if self.stop_training:
                    break
                
                # Learning Rate Decay: Exponential decay
                current_lr = initial_lr * (0.95 ** epoch)
                
                # Train-Set shufflen für bessere Generalisierung
                train_indices_shuffled = np.random.permutation(len(train_boards))
                train_boards_shuffled = train_boards[train_indices_shuffled]
                train_moves_shuffled = train_moves[train_indices_shuffled]
                
                # Training auf Train-Set (train_on_data_set liefert loss + korrekt-Flag für Train-Accuracy)
                train_loss = 0
                train_correct = 0
                n_train = len(train_boards_shuffled)
                for i in range(n_train):
                    loss, correct = train_on_data_set(model, train_boards_shuffled[i], train_moves_shuffled[i], current_lr, weight_decay=0.0)
                    train_loss += loss
                    train_correct += int(correct)
                    
                    # Status-Update nur alle 500 Samples (weniger UI-Overhead)
                    if (i + 1) % 500 == 0:
                        if self.stop_training:
                            break
                        cur_loss = train_loss / (i + 1)
                        cur_acc = train_correct / (i + 1) * 100
                        self.root.after(0, self.status_label.config, 
                                      {"text": f"Epoche {epoch + 1}/{epochs} | {i + 1}/{n_train} | Loss: {cur_loss:.3f} | Train Acc: {cur_acc:.1f}%", 
                                       "foreground": "blue"})
                
                if self.stop_training:
                    break
                
                avg_train_loss = train_loss / n_train
                train_accuracy = train_correct / n_train * 100
                
                # Test: Loss und Accuracy (1 Forward pro Sample, argmax für Accuracy)
                n_test = len(test_boards)
                test_loss = 0
                test_correct = 0
                for i in range(n_test):
                    output = model.forward(test_boards[i])
                    exps = np.exp(output - np.max(output))
                    probs = exps / np.sum(exps)
                    test_loss -= np.sum(test_moves[i] * np.log(probs + 1e-15))
                    if np.argmax(output) == np.argmax(test_moves[i]):
                        test_correct += 1
                
                avg_test_loss = test_loss / n_test
                test_accuracy = test_correct / n_test * 100
                
                self.epoch_history.append(epoch + 1)
                self.loss_history.append(avg_train_loss)
                self.train_accuracy_history.append(train_accuracy)
                self.test_loss_history.append(avg_test_loss)
                self.test_accuracy_history.append(test_accuracy)
                
                self.root.after(0, self.update_status, epoch + 1, epochs, avg_train_loss, train_accuracy, avg_test_loss, test_accuracy)
                self.root.after(0, self.update_chart)
            
            # Bei Stopp-Abruch: keine weiteren Epochen
            if self.stop_training:
                self.root.after(0, self.status_label.config, {"text": f"Training gestoppt nach {len(self.epoch_history)} Epochen.", "foreground": "orange"})
            
            # Modell speichern (auch bei vorzeitigem Stopp)
            self.root.after(0, self.status_label.config, {"text": "Speichere Modell...", "foreground": "blue"})
            os.makedirs("models", exist_ok=True)
            model.save_model("models/")
            
            # Aktuelles Modell speichern für späteres Speichern im models-Ordner
            self.current_model = model
            
            if not self.stop_training:
                ta, tea = self.train_accuracy_history[-1], self.test_accuracy_history[-1]
                self.root.after(0, self.status_label.config, {"text": f"Fertig! Train Acc: {ta:.1f}% | Test Acc: {tea:.1f}% | Train Loss: {self.loss_history[-1]:.4f} | Test Loss: {self.test_loss_history[-1]:.4f}", "foreground": "green"})
            self.root.after(0, self.refresh_models)
            
        except Exception as e:
            self.root.after(0, self.status_label.config, {"text": f"Fehler: {str(e)}", "foreground": "red"})
        finally:
            self.is_training = False
            self.stop_training = False
            self.root.after(0, self.start_button.config, {"state": "normal"})
            self.root.after(0, self.stop_button.config, {"state": "disabled"})
    
    def update_status(self, epoch, total_epochs, train_loss, train_accuracy, test_loss, test_accuracy):
        self.status_label.config(text=f"Epoche {epoch}/{total_epochs} | Train Acc: {train_accuracy:.1f}% | Test Acc: {test_accuracy:.1f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}", foreground="blue")
    
    def refresh_datasets(self):
        """Aktualisiert die Liste der verfügbaren Datasets"""
        self.dataset_paths, self.dataset_names = get_available_datasets()
        if self.dataset_names:
            current = self.dataset_var.get()
            self.dataset_combo['values'] = self.dataset_names
            if current in self.dataset_names:
                self.dataset_var.set(current)
            else:
                self.dataset_var.set(self.dataset_names[0])
        else:
            self.dataset_combo['values'] = []
            self.dataset_var.set("")
    
    def refresh_models(self):
        """Aktualisiert die Liste der verfügbaren Modelle"""
        self.model_paths, self.model_names = get_available_models()
        if self.model_names:
            current = self.model_var.get()
            self.model_combo['values'] = self.model_names
            if current in self.model_names:
                self.model_var.set(current)
            else:
                self.model_var.set(self.model_names[0])
        else:
            self.model_combo['values'] = []
            self.model_var.set("")
        
        # Auch Play-Tab Modelle aktualisieren
        self.play_model_paths, self.play_model_names = get_available_models()
        if hasattr(self, 'play_model_combo'):
            if self.play_model_names:
                current_play = self.play_model_var.get()
                self.play_model_combo['values'] = self.play_model_names
                if current_play in self.play_model_names:
                    self.play_model_var.set(current_play)
                else:
                    self.play_model_var.set(self.play_model_names[0])
            else:
                self.play_model_combo['values'] = []
                self.play_model_var.set("")
    
    def save_current_model(self):
        """Speichert das aktuell trainierte Modell im models-Ordner"""
        if self.current_model is None:
            self.status_label.config(text="Kein Modell zum Speichern verfügbar", foreground="red")
            return
        
        try:
            os.makedirs("models", exist_ok=True)
            filename_date_time = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
            filepath = os.path.join("models", f"model_{filename_date_time}.npz")
            
            weights = {}
            for i, layer in enumerate(self.current_model.layers):
                for j, neuron in enumerate(layer.neurons):
                    weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                    weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
            
            np.savez(filepath, **weights)
            self.status_label.config(text=f"Modell gespeichert: {filepath}", foreground="green")
            self.refresh_models()
        except Exception as e:
            self.status_label.config(text=f"Fehler beim Speichern: {str(e)}", foreground="red")
    
    def test_model(self):
        """Testet ein geladenes Modell auf beiden Test-Methoden (realistisch + zufällig)"""
        if self.is_training:
            return
        
        try:
            num_hidden_layers = int(self.num_layers_var.get())
            hidden_neurons = int(self.hidden_neurons_var.get())
        except ValueError:
            self.status_label.config(text="Fehler: Ungültige Netzwerk-Parameter", foreground="red")
            return
        
        selected_name = self.model_var.get()
        if not selected_name:
            self.status_label.config(text="Kein Modell ausgewählt", foreground="red")
            return
        
        model_path = None
        if selected_name and self.model_paths:
            try:
                idx = self.model_names.index(selected_name)
                model_path = self.model_paths[idx]
            except (ValueError, IndexError):
                self.status_label.config(text="Fehler: Modell nicht gefunden", foreground="red")
                return
        
        self.test_button.config(state="disabled")
        self.status_label.config(text="Lade Modell und teste...", foreground="blue")
        
        def test():
            try:
                model = load_model(model_path, num_hidden_layers, hidden_neurons)
                if model is None:
                    self.root.after(0, self.status_label.config, {"text": "Fehler: Modell konnte nicht geladen werden", "foreground": "red"})
                    return
                
                # Test 1: Realistische Spielverläufe
                self.root.after(0, self.status_label.config, {"text": "Teste Modell auf realistischen Spielverläufen...", "foreground": "blue"})
                realistic_accuracy, realistic_total, realistic_correct = test_on_realistic_games(model, num_test_boards=100)
                
                # Test 2: Zufällige Boards
                self.root.after(0, self.status_label.config, {"text": "Teste Modell auf zufälligen Spielständen...", "foreground": "blue"})
                random_accuracy, random_total, random_correct = test_model_on_random_boards(model, num_test_boards=100)
                
                # Beide Ergebnisse anzeigen
                result_text = (f"Test abgeschlossen!\n"
                              f"Realistische Spiele: {realistic_correct}/{realistic_total} korrekt ({realistic_accuracy:.2f}%)\n"
                              f"Zufällige Boards: {random_correct}/{random_total} korrekt ({random_accuracy:.2f}%)")
                
                self.root.after(0, self.status_label.config, 
                              {"text": result_text, 
                               "foreground": "green"})
            except Exception as e:
                self.root.after(0, self.status_label.config, {"text": f"Fehler beim Testen: {str(e)}", "foreground": "red"})
            finally:
                self.root.after(0, self.test_button.config, {"state": "normal"})
        
        thread = threading.Thread(target=test)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
