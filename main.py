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
from NeuralNetwork.rnn_fast.fast_predict import FastNeuralNetwork, create_fast_layer
from NeuralNetwork.rnn_fast.fast_training import train_batch
from datetime import datetime
import hashlib

def get_available_datasets():
    """Gibt eine Liste aller verfÃ¼gbaren Datasets zurÃ¼ck"""
    dataset_dir = "datasets"
    
    # Suche nach allen .npy Dateien im datasets-Ordner
    pattern = os.path.join(dataset_dir, "*.npy")
    datasets = glob.glob(pattern)
    
    # Sortiere nach Modifikationszeit (neueste zuerst)
    datasets.sort(key=os.path.getmtime, reverse=True)
    
    # Extrahiere nur die Dateinamen fÃ¼r die Anzeige
    dataset_names = [os.path.basename(d) for d in datasets]
    return datasets, dataset_names


def load_dataset(dataset_path=None):
    """LÃ¤dt das Dataset aus training-dataset.

    Es wird keine Symmetrie-Augmentation verwendet; Daten stammen direkt aus
    der Minimax-Generierung und reprÃ¤sentieren reale SpielzustÃ¤nde.

    Args:
        dataset_path: Pfad zum Dataset (None = neuestes Dataset)
    """
    if dataset_path is None:
        datasets, _ = get_available_datasets()
        if not datasets:
            return None
        dataset_path = datasets[0]

    if not os.path.exists(dataset_path):
        return None

    data = np.load(dataset_path)
    boards = data[:, :9].astype(float)
    
    if data.shape[1] == 18:
        # Neues Multi-Hot Format
        moves = data[:, 9:18].astype(int)
    else:
        # Altes Format: Letzter Wert = bester Zug
        moves_idx = data[:, 9].astype(int)
        moves = np.zeros((len(moves_idx), 9))
        moves[np.arange(len(moves_idx)), moves_idx] = 1

    return boards, moves

def get_available_models():
    """Gibt eine Liste aller verfÃ¼gbaren Modelle zurÃ¼ck (nur aus dem models-Ordner)"""
    models_dir = "models"
    models_pattern = os.path.join(models_dir, "*.npz")
    
    all_models = glob.glob(models_pattern)
    
    # Sortiere nach Modifikationszeit (neueste zuerst)
    all_models.sort(key=os.path.getmtime, reverse=True)
    
    # Extrahiere nur die Dateinamen fÃ¼r die Anzeige
    model_names = [os.path.basename(m) for m in all_models]
    return all_models, model_names

def load_model(model_path):
    """LÃ¤dt ein gespeichertes Modell aus einer .npz Datei und rekonstruiert automatisch
    die Schichten.

    Die Datei muss Gewichte/Biases mit SchlÃ¼sseln der Form
    ``layer_{i}_neuron_{j}_weights``/``_bias`` enthalten.
    """
    if not os.path.exists(model_path):
        return None
    
    data = np.load(model_path)
    layers = []
    layer_idx = 0
    # solange es Neuronen in layer_{layer_idx} gibt, weiterbauen
    while True:
        prefix = f'layer_{layer_idx}_neuron_'
        # finde alle weight-schlÃ¼ssel fÃ¼r diesen layer
        keys = [k for k in data.keys() if k.startswith(prefix) and k.endswith('_weights')]
        if not keys:
            break
        # sortiere numerisch nach neuron_{j} (nicht lexikographisch!)
        keys.sort(key=lambda k: int(k.split('_neuron_')[1].split('_')[0]))
        neurons = []
        for key in keys:
            # extrahiere j aus dem key, aber wir brauchen es nicht
            base = key.rsplit('_', 1)[0]  # layer_i_neuron_j
            weight = data[f"{base}_weights"]
            bias = data[f"{base}_bias"]
            neurons.append(Neuron(weight, bias))
        layers.append(Layer(neurons))
        layer_idx += 1
    model = NeuralNetwork(layers)
    return model

def generate_random_board():
    """Generiert einen zufÃ¤lligen gÃ¼ltigen Tic-Tac-Toe Spielstand"""
    board = np.zeros(9, dtype=int)
    
    # ZufÃ¤llige Anzahl von ZÃ¼gen (1-8, da mindestens ein Feld frei bleiben muss)
    num_moves = np.random.randint(1, 9)
    
    # ZufÃ¤llige Reihenfolge der Felder
    positions = np.random.permutation(9)
    
    # FÃ¼lle das Board abwechselnd mit 1 und -1
    for i in range(num_moves):
        pos = positions[i]
        player = 1 if i % 2 == 0 else -1
        board[pos] = player
    
    return board

def is_valid_board(board):
    """PrÃ¼ft ob ein Board gÃ¼ltig ist (nicht gewonnen, nicht voll)"""
    # PrÃ¼fe auf Gewinner
    b = board.reshape(3, 3)
    lines = [
        b[0], b[1], b[2],
        b[:, 0], b[:, 1], b[:, 2],
        b.diagonal(), np.fliplr(b).diagonal()
    ]
    
    for line in lines:
        if np.all(line == 1) or np.all(line == -1):
            return False
    
    # PrÃ¼fe ob voll
    if np.all(board != 0):
        return False
    
    return True

def generate_test_boards(num_boards=100):
    """Generiert zufÃ¤llige gÃ¼ltige Test-Boards"""
    boards = []
    attempts = 0
    max_attempts = num_boards * 10
    
    while len(boards) < num_boards and attempts < max_attempts:
        board = generate_random_board()
        if is_valid_board(board):
            boards.append(board)
        attempts += 1
    
    return np.array(boards)

def validate_dataset(dataset_path):
    """
    Validiert die QualitÃ¤t eines Datasets und gibt eine Statistik aus.
    
    PrÃ¼ft auf:
    1. Illegale Labels (Zug nicht auf leerem Feld)
    2. Invalide Boards (falsche Steinzahlen)
    3. Kanonisierung (Boards aus Sicht des Spielers)
    
    Returns:
        dict mit Statistiken
    """
    if not os.path.exists(dataset_path):
        return None
    
    data = np.load(dataset_path)
    boards = data[:, :9].astype(int)
    moves = data[:, 9].astype(int)
    
    illegal_labels = 0
    invalid_boards = 0
    
    for i, (board, move) in enumerate(zip(boards, moves)):
        # Check: Ist der Label auf einem leeren Feld?
        if board[move] != 0:
            illegal_labels += 1
        
        # Check: Ist das Board kanonisiert? (nur Werte -1, 0, +1)
        if not np.all(np.isin(board, [-1, 0, 1])):
            invalid_boards += 1
    
    stats = {
        "total_samples": len(boards),
        "illegal_labels": illegal_labels,
        "illegal_labels_pct": 100 * illegal_labels / len(boards) if len(boards) > 0 else 0,
        "invalid_boards": invalid_boards,
        "invalid_boards_pct": 100 * invalid_boards / len(boards) if len(boards) > 0 else 0,
        "quality": "OK" if illegal_labels == 0 and invalid_boards == 0 else "PROBLEMATIC"
    }
    
    return stats

def test_model_on_random_boards(model, num_test_boards=100):
    """Testet ein Modell auf zufÃ¤lligen SpielstÃ¤nden"""
    test_boards = generate_test_boards(num_test_boards)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    find_all_best_moves_for_player = dataset_module.find_all_best_moves_for_player
    get_current_player = dataset_module.get_current_player
    
    correct_predictions = 0
    total_tests = 0
    
    for board in test_boards:
        if not is_valid_board(board):
            continue
        
        player = get_current_player(board)
        optimal_moves = find_all_best_moves_for_player(board, player)
        if not optimal_moves:
            continue
        
        # Kanonisierung fÃ¼r Modell
        canonical_board = dataset_module.canonicalize_board(board, player)
        
        # Legal mask
        legal_mask = (canonical_board == 0).astype(float)
        
        # Modell-Vorhersage (auf kanonisiertem Board mit Maskierung)
        predicted_move, _ = model.predict(canonical_board.astype(float), legal_mask)
        
        if predicted_move in optimal_moves:
            correct_predictions += 1
        total_tests += 1
    
    accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    return accuracy, total_tests, correct_predictions

def test_on_realistic_games(model, num_test_boards=100):
    """Testet ein Modell auf realistischen SpielverlÃ¤ufen (wie im Training)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    find_all_best_moves_for_player = dataset_module.find_all_best_moves_for_player
    get_next_boards = dataset_module.get_next_boards
    check_winner = dataset_module.check_winner
    
    # Generiere realistische SpielverlÃ¤ufe (BFS) - speichere (board, player)
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
        optimal_moves = find_all_best_moves_for_player(board, player)
        if not optimal_moves:
            continue
        
        # Kanonisierung fÃ¼r Modell
        canonical_board = dataset_module.canonicalize_board(board, player)
        
        # Legal mask fÃ¼r kanonisiertes Board
        legal_mask = (canonical_board == 0).astype(float)
        
        # Modell-Vorhersage (auf kanonisiertem Board mit Maskierung)
        predicted_move, _ = model.predict(canonical_board.astype(float), legal_mask)
        
        if predicted_move in optimal_moves:
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
        self.training_active = False  # For overview tab
        self.stop_training = False
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.epoch_history = []
        self.current_model = None
        
        # Default config values (for overview tab before setup_training_tab)
        self.epochs_var = tk.StringVar(value="300")
        self.lr_var = tk.StringVar(value="0.005")
        self.weight_decay_var = tk.StringVar(value="0.0005")
        self.num_layers_var = tk.StringVar(value="3")
        self.neurons_var = tk.StringVar(value="128")
        
        # Dataset-Generierung Flags
        self.gen_stop_event = threading.Event()
        self.gen_pause_event = threading.Event()
        
        # Game state fÃ¼r Play Tab
        self.game_board = np.zeros(9, dtype=int)
        self.game_model = None
        self.game_active = False
        self.player_turn = True  # True = Spieler (X), False = Modell (O)
        self.player_symbol = 1  # Spieler ist X (1)
        self.model_symbol = -1  # Modell ist O (-1)


        # Analyse/Hinweise fÃ¼r Spiel-Tab
        self.game_dataset_path = None
        self.game_dataset_lookup = {}
        self.game_find_best_move_for_player = None
        self.game_find_all_best_moves_for_player = None
        self.game_get_current_player = None

        
        self.setup_ui()
        
    def setup_ui(self):
        # Notebook fÃ¼r Tabs erstellen
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 0: Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        
        # Tab 1: Dataset-Generator
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="Dataset")
        
        # Tab 2: Testing
        self.tests_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tests_tab, text="Testing")
        
        # Tab 3: Spielen
        self.play_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.play_tab, text="Spielen")
        
        # Setup der einzelnen Tabs
        self.setup_training_tab()
        self.setup_dataset_tab()
        self.setup_tests_tab()
        self.setup_play_tab()
    

    
    def setup_training_tab(self):
        # ===== CONFIG FRAME =====
        config_frame = ttk.LabelFrame(self.training_tab, text="Training Konfiguration", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # ROW 0: Epochs, LR, Weight Decay
        ttk.Label(config_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="300")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Learning Rate:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.lr_var = tk.StringVar(value="0.005")
        ttk.Entry(config_frame, textvariable=self.lr_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Weight Decay:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.weight_decay_var = tk.StringVar(value="0.0005")
        ttk.Entry(config_frame, textvariable=self.weight_decay_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # ROW 1: Hidden Layers, Neurons
        ttk.Label(config_frame, text="Hidden Layers:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_layers_var = tk.StringVar(value="3")
        self.num_layers_entry = ttk.Entry(config_frame, textvariable=self.num_layers_var, width=10)
        self.num_layers_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Neurons/Layer:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.hidden_neurons_var = tk.StringVar(value="128")
        self.hidden_neurons_entry = ttk.Entry(config_frame, textvariable=self.hidden_neurons_var, width=10)
        self.hidden_neurons_entry.grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Early Stop Patience:").grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        self.patience_var = tk.StringVar(value="30")
        ttk.Entry(config_frame, textvariable=self.patience_var, width=10).grid(row=1, column=5, padx=5, pady=5)
        
        # ROW 2: Dataset Selection
        ttk.Label(config_frame, text="Dataset:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.dataset_paths, self.dataset_names = get_available_datasets()
        self.dataset_var = tk.StringVar()
        if self.dataset_names:
            self.dataset_var.set(self.dataset_names[0])
        self.dataset_combo = ttk.Combobox(config_frame, textvariable=self.dataset_var, values=self.dataset_names, 
                                         state="readonly", width=30)
        self.dataset_combo.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        # Auto-refresh dropdown when datasets change
        self.dataset_var.trace_add("write", lambda *args: self.refresh_datasets())

        self.split_unique_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Unique-Board Split", variable=self.split_unique_var).grid(
            row=2, column=3, padx=5, pady=5, sticky=tk.W
        )

        self.same_train_test_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Train=Test (Debug)", variable=self.same_train_test_var).grid(
            row=2, column=4, columnspan=2, padx=5, pady=5, sticky=tk.W
        )
        
        # ROW 3: Buttons
        self.start_button = ttk.Button(config_frame, text="START TRAINING", command=self.start_training)
        self.start_button.grid(row=3, column=0, padx=5, pady=5, sticky=tk.EW)
        self.start_button.config(style="Accent.TButton" if hasattr(ttk.Style(), "Accent.TButton") else "")
        
        self.stop_button = ttk.Button(config_frame, text="STOP", command=self.stop_training_request, state="disabled")
        self.stop_button.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        
        self.finetune_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Finetune Mode", variable=self.finetune_var).grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.finetune_var.trace_add("write", lambda *args: self._on_finetune_toggle())
        
        # ROW 4: Status
        self.status_label = ttk.Label(config_frame, text="Ready", foreground="green", font=("", 10, "bold"))
        self.status_label.grid(row=4, column=0, columnspan=6, pady=10, sticky=tk.W)
        
        # ROW 5: Save model controls
        ttk.Label(config_frame, text="Save as:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.save_name_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.save_name_var, width=20).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(config_frame, text="SAVE MODEL", command=self.save_current_model).grid(row=5, column=2, padx=5, pady=5, sticky=tk.EW)
        
        # ===== PROGRESS FRAME =====
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Progress Bar
        ttk.Label(progress_frame, text="Epoch Progress:").pack(anchor=tk.W)
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_bar = ttk.Progressbar(progress_frame, variable=self.train_progress_var, maximum=100, length=400)
        self.train_progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_text = ttk.Label(progress_frame, text="Epoch 0/0", foreground="blue")
        self.progress_text.pack(anchor=tk.W)
        
        # ===== CHART & METRICS CONTAINER =====
        chart_container = ttk.Frame(self.training_tab)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Chart (smaller)
        self.setup_chart(chart_container)
        
        # Right: Metrics panel
        metrics_frame = ttk.LabelFrame(chart_container, text="Aktuelle Metriken", padding="10")
        metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Metrics display with monospace font for alignment
        self.metrics_display = tk.Text(metrics_frame, height=20, width=25, state=tk.DISABLED, font=("Courier", 9))
        self.metrics_display.pack(fill=tk.BOTH, expand=True)
        
        self._update_metrics_display()

    def _on_finetune_toggle(self):
        """Aktualisiert den Status der Architekturfelder je nach Checkbox."""
        enabled = not self.finetune_var.get()
        state = "normal" if enabled else "disabled"
        self.num_layers_entry.config(state=state)
        self.hidden_neurons_entry.config(state=state)
    
    def _update_mode_description(self, *args):
        """Aktualisiert die Modus-Beschreibung."""
        mode = self.new_dataset_mode_tab_var.get()
        descriptions = {
            "tree": "Tree: Tiefensuche, viele Endgame-Positionen",
            "selfplay": "Selfplay: Realistische Partien, ausgewogen",
            "optimal": "Optimal: Systematisch, ausgewogen, beste QualitÃ¤t"
        }
        self.mode_desc_label.config(text=descriptions.get(mode, ""))
        
    def setup_dataset_tab(self):
        """Erstellt UI zum Generieren von DatensÃ¤tzen mit erweiterten Funktionen"""
        # ===== OBERER BEREICH: Generator =====
        generator_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Generator", padding="10")
        generator_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Row 0: GrÃ¶ÃŸe & Modus
        ttk.Label(generator_frame, text="GrÃ¶ÃŸe:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.new_dataset_size_tab_var = tk.StringVar(value="1000")
        ttk.Entry(generator_frame, textvariable=self.new_dataset_size_tab_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(generator_frame, text="Modus:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.new_dataset_mode_tab_var = tk.StringVar(value="optimal")
        ttk.Combobox(generator_frame, textvariable=self.new_dataset_mode_tab_var,
                     values=["tree", "selfplay", "optimal"], state="readonly", width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Modus-Beschreibung
        self.mode_desc_label = ttk.Label(generator_frame, text="Optimal: Systematisch, ausgewogen, beste QualitÃ¤t", foreground="blue")
        self.mode_desc_label.grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        # Update description when mode changes
        self.new_dataset_mode_tab_var.trace_add("write", self._update_mode_description)
        
        ttk.Label(generator_frame, text="Name:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.dataset_name_var = tk.StringVar()
        ttk.Entry(generator_frame, textvariable=self.dataset_name_var, width=20).grid(row=0, column=5, padx=5, pady=5)
        
        # Row 1: Start/Pause/Abort Buttons
        self.gen_start_button = ttk.Button(generator_frame, text="Start", command=self.start_generate_dataset)
        self.gen_start_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.gen_pause_button = ttk.Button(generator_frame, text="Pausieren", command=self.pause_generate_dataset, state=tk.DISABLED)
        self.gen_pause_button.grid(row=1, column=1, padx=5, pady=5)
        
        self.gen_abort_button = ttk.Button(generator_frame, text="Abbrechen", command=self.abort_generate_dataset, state=tk.DISABLED)
        self.gen_abort_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Row 2: Progress Bar
        self.gen_progress_var = tk.DoubleVar()
        self.gen_progress_bar = ttk.Progressbar(generator_frame, variable=self.gen_progress_var, maximum=100, length=300)
        self.gen_progress_bar.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW)
        
        self.gen_progress_label = ttk.Label(generator_frame, text="0/0 samples")
        self.gen_progress_label.grid(row=2, column=3, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        # Row 3: Status
        self.gen_status_label = ttk.Label(generator_frame, text="Bereit", foreground="green")
        self.gen_status_label.grid(row=3, column=0, columnspan=6, pady=5, sticky=tk.W)
        
        # ===== UNTERER BEREICH: Dataset-Ãœbersicht =====
        overview_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Ãœbersicht", padding="10")
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview fÃ¼r Dataset-Ãœbersicht
        tree_columns = ("Name", "GrÃ¶ÃŸe", "Modus", "Samples", "Status")
        self.dataset_tree = ttk.Treeview(overview_frame, columns=tree_columns, height=10)
        self.dataset_tree.column("#0", width=0, stretch=tk.NO)
        self.dataset_tree.column("Name", anchor=tk.W, width=200)
        self.dataset_tree.column("GrÃ¶ÃŸe", anchor=tk.CENTER, width=100)
        self.dataset_tree.column("Modus", anchor=tk.CENTER, width=80)
        self.dataset_tree.column("Samples", anchor=tk.CENTER, width=80)
        self.dataset_tree.column("Status", anchor=tk.CENTER, width=100)
        
        self.dataset_tree.heading("#0", text="", anchor=tk.W)
        self.dataset_tree.heading("Name", text="Name", anchor=tk.W)
        self.dataset_tree.heading("GrÃ¶ÃŸe", text="GrÃ¶ÃŸe (Bytes)", anchor=tk.CENTER)
        self.dataset_tree.heading("Modus", text="Modus", anchor=tk.CENTER)
        self.dataset_tree.heading("Samples", text="Samples", anchor=tk.CENTER)
        self.dataset_tree.heading("Status", text="Status", anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscroll=scrollbar.set)
        
        self.dataset_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initiale Ãœbersicht
        self.refresh_dataset_overview()

    def setup_chart(self, parent_frame):
        self.fig = Figure(figsize=(4.5, 4), dpi=100)
        self.ax_loss = self.fig.add_subplot(111)
        self.ax_loss.set_xlabel("Epoche", fontsize=9)
        self.ax_loss.set_ylabel("Loss", color='b', fontsize=9)
        self.ax_loss.tick_params(axis='y', labelcolor='b', labelsize=8)
        self.ax_loss.tick_params(axis='x', labelsize=8)
        self.ax_loss.grid(True, alpha=0.3)
        
        # Zweite y-Achse fÃ¼r Accuracy
        self.ax_acc = self.ax_loss.twinx()
        self.ax_acc.set_ylabel("Accuracy (%)", color='r', fontsize=9)
        self.ax_acc.tick_params(axis='y', labelcolor='r', labelsize=8)
        
        self.fig.suptitle("Training Progress", fontsize=10)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def update_chart(self):
        if len(self.epoch_history) > 0:
            self.ax_loss.clear()
            self.ax_acc.clear()
            
            # Only plot essential data
            if len(self.loss_history) > 0:
                self.ax_loss.plot(self.epoch_history, self.loss_history, 'b-', linewidth=1.5, label='Train Loss')
            if len(self.test_loss_history) > 0:
                self.ax_loss.plot(self.epoch_history, self.test_loss_history, 'b--', linewidth=1.5, label='Test Loss')
            
            if len(self.test_accuracy_history) > 0:
                self.ax_acc.plot(self.epoch_history, self.test_accuracy_history, 'r-', linewidth=1.5, label='Test Acc')
            
            self.ax_loss.set_xlabel("Epoche", fontsize=9)
            self.ax_loss.set_ylabel("Loss", color='b', fontsize=9)
            self.ax_loss.tick_params(axis='y', labelcolor='b', labelsize=8)
            self.ax_loss.tick_params(axis='x', labelsize=8)
            self.ax_loss.grid(True, alpha=0.3)
            
            self.ax_acc.set_ylabel("Accuracy (%)", color='r', fontsize=9)
            self.ax_acc.tick_params(axis='y', labelcolor='r', labelsize=8)
            
            lines1, labels1 = self.ax_loss.get_legend_handles_labels()
            lines2, labels2 = self.ax_acc.get_legend_handles_labels()
            if lines1 or lines2:
                self.ax_loss.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
            
            self.fig.suptitle("Loss & Accuracy", fontsize=10)
            self.canvas.draw()
            
            # Update metrics display
            if hasattr(self, '_update_metrics_display'):
                self._update_metrics_display()
    
    def setup_play_tab(self):
        """Erstellt die UI fÃ¼r das Spielen gegen das Modell"""
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
        # Auto-refresh models dropdown
        self.play_model_var.trace_add("write", lambda *args: self._refresh_play_models())
        
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


        self.game_hint_label = ttk.Label(
            model_frame,
            text="Hinweise: Minimax wird nach jedem Zug angezeigt.",
            foreground="#555555"
        )
        self.game_hint_label.grid(row=2, column=0, columnspan=8, pady=(0, 8))

        self.show_minimax_hints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            model_frame,
            text="Minimax-Hinweise anzeigen",
            variable=self.show_minimax_hints_var,
            command=self.update_board_hints
        ).grid(row=3, column=0, columnspan=3, padx=5, pady=(0, 8), sticky=tk.W)

        
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

        # Minimax-/Dataset-Berater initialisieren (fÃ¼r Hinweise + Sicherheitsnetz)
        self.initialize_game_advisors()
    

    def setup_tests_tab(self):
        """Erstellt die UI fÃ¼r das Testen von Modellen gegen verschiedene Gegner"""
        # Modell-Auswahl
        model_frame = ttk.LabelFrame(self.tests_tab, text="Modell & Einstellungen", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Modell:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.test_model_paths, self.test_model_names = get_available_models()
        self.test_model_var = tk.StringVar()
        if self.test_model_names:
            self.test_model_var.set(self.test_model_names[0])
        self.test_model_combo = ttk.Combobox(model_frame, textvariable=self.test_model_var, 
                                            values=self.test_model_names, state="readonly", width=30)
        self.test_model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        # Auto-refresh models dropdown
        self.test_model_var.trace_add("write", lambda *args: self._refresh_test_models())
        
        ttk.Label(model_frame, text="Anzahl Spiele:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.test_num_games_var = tk.StringVar(value="10")
        ttk.Spinbox(model_frame, from_=1, to=100, textvariable=self.test_num_games_var, width=5).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Test-Typ:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.test_type_var = tk.StringVar(value="Model vs Minimax")
        test_types = ["Model vs Minimax", "Model vs Random", "Fair Accuracy"]
        ttk.Combobox(model_frame, textvariable=self.test_type_var, values=test_types, state="readonly", width=18).grid(row=0, column=5, padx=5, pady=5)
        
        # Buttons
        self.test_start_button = ttk.Button(model_frame, text="Tests starten", command=self.start_tests)
        self.test_start_button.grid(row=0, column=6, padx=5, pady=5)
        
        # Status
        self.test_status_label = ttk.Label(model_frame, text="Bereit", foreground="green")
        self.test_status_label.grid(row=1, column=0, columnspan=7, pady=5, sticky=tk.W)
        
        # Progress Bar
        self.test_progress_var = tk.DoubleVar()
        self.test_progress_bar = ttk.Progressbar(model_frame, variable=self.test_progress_var, maximum=100, length=300)
        self.test_progress_bar.grid(row=2, column=0, columnspan=7, padx=5, pady=5, sticky=tk.EW)
        
        # Results Frame
        results_frame = ttk.LabelFrame(self.tests_tab, text="Ergebnisse", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results Text mit Scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.test_results_text = tk.Text(results_frame, height=20, width=80, yscrollcommand=scrollbar.set)
        self.test_results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.test_results_text.yview)
        
        # Test werden spÃ¤ter initialisiert
        self.test_running = False

    def initialize_game_advisors(self):
        """LÃ¤dt Dataset + Minimax-Helfer fÃ¼r Zug-Hinweise im Spiel-Tab."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
            dataset_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_module)

            self.game_find_best_move_for_player = dataset_module.find_best_move_for_player
            self.game_find_all_best_moves_for_player = dataset_module.find_all_best_moves_for_player
            self.game_get_current_player = dataset_module.get_current_player
        except Exception as e:
            self.game_find_best_move_for_player = None
            self.game_find_all_best_moves_for_player = None
            self.game_get_current_player = None
            self.game_hint_label.config(text=f"Hinweis-Fehler (Minimax): {str(e)}", foreground="red")
            return

        self.game_dataset_path = None
        self.game_dataset_lookup = {}

    def update_board_hints(self):
        """Markiert freie Felder gemÃ¤ÃŸ Dataset-Label und Minimax-OptimalzÃ¼gen."""
        # Grundfarben fÃ¼r freie Felder zurÃ¼cksetzen
        for i in range(9):
            if self.game_board[i] == 0:
                self.board_buttons[i].config(bg="SystemButtonFace")

        # Ohne Helfer keine Hinweise
        if self.game_get_current_player is None or self.game_find_all_best_moves_for_player is None:
            return

        board_int = self.game_board.astype(int)
        current_player = self.game_get_current_player(board_int)

        # Minimax: alle optimalen ZÃ¼ge (togglebar)
        show_minimax = bool(self.show_minimax_hints_var.get()) if hasattr(self, 'show_minimax_hints_var') else True
        minimax_moves = self.game_find_all_best_moves_for_player(board_int, current_player) if show_minimax else []
        minimax_set = set(minimax_moves)

        # Farbcode:
        # - orange: Minimax-optimal
        # Nur fÃ¼r freie Felder markieren
        for i in range(9):
            if self.game_board[i] != 0:
                continue
            in_minimax = i in minimax_set
            if show_minimax and in_minimax:
                self.board_buttons[i].config(bg="#ffd580")

        turn_text = "X am Zug" if current_player == 1 else "O am Zug"
        if show_minimax:
            minimax_text = f"Minimax-optimal: {sorted(minimax_moves)}"
            text = f"{turn_text} | {minimax_text} | Farbe: Orange=Minimax"
        else:
            text = f"{turn_text} | Minimax-Hinweise: AUS"

        self.game_hint_label.config(text=text, foreground="#333333")


    def load_game_model(self):
        """LÃ¤dt das ausgewÃ¤hlte Modell fÃ¼r das Spiel"""
        selected_name = self.play_model_var.get()
        if not selected_name:
            self.game_status_label.config(text="Kein Modell ausgewÃ¤hlt", foreground="red")
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
            self.game_model = load_model(model_path)
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
        
        # Board zurÃ¼cksetzen
        self.game_board = np.zeros(9, dtype=int)
        self.game_active = True
        self.player_turn = True  # Spieler beginnt
        
        # Buttons zurÃ¼cksetzen
        for btn in self.board_buttons:
            btn.config(text="", state=tk.NORMAL, bg="SystemButtonFace")
        
        self.game_status_label.config(text="Dein Zug (X). Klicke auf ein Feld.", foreground="blue")
        self.update_board_hints()

    
    def make_move(self, position):
        """Spieler macht einen Zug"""
        if not self.game_active or not self.player_turn:
            return
        
        if self.game_board[position] != 0:
            return  # Feld bereits belegt
        
        # Spieler-Zug
        self.game_board[position] = self.player_symbol
        self.board_buttons[position].config(text="X", state=tk.DISABLED, bg="lightblue")
        self.update_board_hints()
        
        # PrÃ¼fe auf Gewinner oder Unentschieden
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
        self.root.after(100, self.model_move)  # Kurze VerzÃ¶gerung fÃ¼r bessere UX

    def find_immediate_winning_move(self, player_symbol):
        """Finde einen Zug, der sofort gewinnt (oder None)."""
        empty_positions = np.where(self.game_board == 0)[0]
        for pos in empty_positions:
            self.game_board[pos] = player_symbol
            winner = self.check_winner()
            self.game_board[pos] = 0
            if winner == player_symbol:
                return int(pos)
        return None
    
    def model_move(self):
        """Modell macht einen Zug"""
        if not self.game_active or self.player_turn:
            return
        
        # Modell-Vorhersage
        try:
            # 1) Taktik-Sicherheitsnetz: sofort gewinnen, falls mÃ¶glich
            winning_move = self.find_immediate_winning_move(self.model_symbol)
            if winning_move is not None:
                predicted_move = winning_move
            else:
                # 2) Taktik-Sicherheitsnetz: sofortigen Gegner-Sieg blocken
                blocking_move = self.find_immediate_winning_move(self.player_symbol)
                if blocking_move is not None:
                    predicted_move = blocking_move
                else:
                    # 3) Sonst NN-Vorhersage
                    # Kanonisierung: Aus Sicht des Modells (O = -1)
                    player_to_move = self.model_symbol  # -1
                    canonical_board = self.game_board * player_to_move  # -1 -> +1, 1 -> -1, 0 -> 0
                    
                    # Legal mask: freie Felder
                    legal_mask = (canonical_board == 0).astype(float)
                    
                    predicted_move, _ = self.game_model.predict(canonical_board.astype(float), legal_mask)
            
            if predicted_move is None:
                self.end_game(0)  # Unentschieden
                return

            if self.game_board[predicted_move] != 0:
                # Fallback auf ersten legalen Zug falls NN unerwartet illegal war
                legal_positions = np.where(self.game_board == 0)[0]
                if len(legal_positions) == 0:
                    self.end_game(0)
                    return
                predicted_move = int(legal_positions[0])

            # 4) Sicherheitsnetz: Falls verfÃ¼gbar, auf Minimax-optimalen Zug korrigieren
            if self.game_get_current_player is not None and self.game_find_all_best_moves_for_player is not None:
                current_player = self.game_get_current_player(self.game_board.astype(int))
                minimax_moves = self.game_find_all_best_moves_for_player(self.game_board.astype(int), current_player)
                if minimax_moves and predicted_move not in minimax_moves:
                    predicted_move = int(minimax_moves[0])
            
            # Modell-Zug
            self.game_board[predicted_move] = self.model_symbol
            self.board_buttons[predicted_move].config(text="O", state=tk.DISABLED, bg="lightcoral")
            
            # PrÃ¼fe auf Gewinner oder Unentschieden
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
            self.update_board_hints()

            
        except Exception as e:
            self.game_status_label.config(text=f"Fehler beim Modell-Zug: {str(e)}", foreground="red")
            self.game_active = False
    
    def check_winner(self):
        """PrÃ¼ft ob es einen Gewinner gibt. Gibt 1 (Spieler), -1 (Modell) oder 0 (kein Gewinner) zurÃ¼ck"""
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
            self.game_status_label.config(text="ðŸŽ‰ Du hast gewonnen!", foreground="green")
            # Gewinnende Linie hervorheben
            self.highlight_winning_line()
        elif winner == -1:
            self.game_status_label.config(text="ðŸ¤– Modell hat gewonnen!", foreground="red")
            self.highlight_winning_line()
        else:
            self.game_status_label.config(text="ðŸ¤ Unentschieden!", foreground="orange")

    
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
            weight_decay = float(self.weight_decay_var.get())
            num_hidden_layers = int(self.num_layers_var.get())
            hidden_neurons = int(self.hidden_neurons_var.get())
            patience = int(self.patience_var.get())
            
            if num_hidden_layers < 1:
                raise ValueError("Hidden Layers muss >= 1 sein")
            if hidden_neurons < 1:
                raise ValueError("Neurons muss >= 1 sein")
        except ValueError as e:
            self.status_label.config(text=f"ERROR: {str(e)}", foreground="red")
            return
        
        continue_flag = self.finetune_var.get()
        if continue_flag:
            selected = self.model_var.get()
            if not selected:
                self.status_label.config(text="ERROR: Kein Modell ausgewÃ¤hlt", foreground="red")
                return
            try:
                idx = self.model_names.index(selected)
                model_path = self.model_paths[idx]
                self.current_model = load_model(model_path)
                if self.current_model is None:
                    raise ValueError("Modell laden fehlgeschlagen")
            except Exception as e:
                self.status_label.config(text=f"ERROR: {str(e)}", foreground="red")
                return
        else:
            self.current_model = None
        
        self.is_training = True
        self.stop_training = False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.loss_history = []
        self.train_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.epoch_history = []
        
        thread = threading.Thread(target=self.train_model, args=(epochs, learning_rate, weight_decay, num_hidden_layers, hidden_neurons, patience, continue_flag))
        thread.daemon = True
        thread.start()
    
    def stop_training_request(self):
        """Setzt das Stopp-Flag, damit die Trainingsschleife abbricht"""
        self.stop_training = True
    
    def train_model(self, epochs, learning_rate, weight_decay, num_hidden_layers, hidden_neurons, patience, continue_flag=False):
        def _save_fast_model(fast_model, path):
            data = {}
            for li, layer in enumerate(fast_model.layers):
                num_neurons = layer.weights.shape[1]
                for j in range(num_neurons):
                    data[f"layer_{li}_neuron_{j}_weights"] = layer.weights[:, j]
                    data[f"layer_{li}_neuron_{j}_bias"] = layer.bias[j]
            np.savez(path, **data)

        def _to_fast_model(classic_model):
            fast_layers = []
            num_layers = len(classic_model.layers)
            for li, cl in enumerate(classic_model.layers):
                input_size = len(cl.neurons[0].weights)
                num_neurons = len(cl.neurons)
                activation = "linear" if li == (num_layers - 1) else "tanh"
                fl = create_fast_layer(input_size, num_neurons, activation=activation)
                fl.weights = np.column_stack([n.weights for n in cl.neurons])
                fl.bias = np.array([n.bias for n in cl.neurons], dtype=float)
                fast_layers.append(fl)
            return FastNeuralNetwork(fast_layers)

        try:
            self.root.after(0, lambda: self.status_label.config(text="Loading dataset...", foreground="blue"))

            selected_name = self.dataset_var.get()
            dataset_path = None
            if selected_name and self.dataset_paths:
                try:
                    idx = self.dataset_names.index(selected_name)
                    dataset_path = self.dataset_paths[idx]
                except (ValueError, IndexError):
                    dataset_path = None

            data = np.load(dataset_path) if dataset_path and os.path.exists(dataset_path) else None
            if data is None:
                self.root.after(0, lambda: self.status_label.config(text="ERROR: Dataset not found!", foreground="red"))
                return

            boards = data[:, :9].astype(float)
            if data.shape[1] == 18:
                moves = data[:, 9:18].astype(int)
            else:
                move_idx = data[:, 9].astype(int)
                moves = np.zeros((len(move_idx), 9), dtype=int)
                moves[np.arange(len(move_idx)), move_idx] = 1

            use_same = bool(self.same_train_test_var.get()) if hasattr(self, 'same_train_test_var') else False
            split_unique = bool(self.split_unique_var.get()) if hasattr(self, 'split_unique_var') else True

            if use_same:
                train_boards = boards.copy()
                train_moves = moves.copy()
                test_boards = boards.copy()
                test_moves = moves.copy()
            elif split_unique:
                unique_boards, unique_indices = np.unique(boards, axis=0, return_index=True)
                np.random.shuffle(unique_indices)
                split_idx = int(len(unique_indices) * 0.8)
                train_indices = np.sort(unique_indices[:split_idx])
                test_indices = np.sort(unique_indices[split_idx:])
                train_boards, train_moves = boards[train_indices], moves[train_indices]
                test_boards, test_moves = boards[test_indices], moves[test_indices]
            else:
                idx = np.random.permutation(len(boards))
                split_idx = int(len(idx) * 0.8)
                train_boards, train_moves = boards[idx[:split_idx]], moves[idx[:split_idx]]
                test_boards, test_moves = boards[idx[split_idx:]], moves[idx[split_idx:]]

            n_train = len(train_boards)
            n_test = len(test_boards)
            self.root.after(0, lambda: self.status_label.config(text=f"Dataset loaded: {n_train} train / {n_test} test", foreground="blue"))

            # Modell erstellen / finetune
            if continue_flag and self.current_model is not None:
                model = _to_fast_model(self.current_model)
                self.root.after(0, lambda: self.status_label.config(text="Finetune mode: Model loaded", foreground="blue"))
            else:
                layers = []
                prev = 9
                for _ in range(num_hidden_layers):
                    layers.append(create_fast_layer(prev, hidden_neurons, activation="relu"))
                    prev = hidden_neurons
                layers.append(create_fast_layer(prev, 9, activation="linear"))
                model = FastNeuralNetwork(layers)

            is_multi_hot = bool(np.any(np.sum(train_moves, axis=1) > 1))

            cfg_text = f"layers-{num_hidden_layers}_neurons-{hidden_neurons}_lr-{learning_rate}_batch-64"
            run_hash = hashlib.md5(cfg_text.encode()).hexdigest()[:6]
            run_name = f"model_ui_{cfg_text}_{run_hash}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            best_model_path = os.path.join("models", f"{run_name}.npz")
            os.makedirs("models", exist_ok=True)

            best_test_acc = -1.0
            epochs_no_improve = 0

            for epoch in range(epochs):
                if self.stop_training:
                    break

                # Train (batch)
                p = np.random.permutation(n_train)
                X_train = train_boards[p]
                Y_train = train_moves[p]

                train_loss_sum = 0.0
                train_correct = 0
                batch_size = 64
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    Xb = X_train[start:end]
                    Yb = Y_train[start:end]
                    masks = (Xb == 0).astype(float)
                    bl, bc = train_batch(
                        model,
                        Xb,
                        Yb,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        legal_masks_batch=masks,
                        multi_hot=is_multi_hot,
                    )
                    train_loss_sum += bl
                    train_correct += bc

                avg_train_loss = train_loss_sum / n_train
                train_acc = 100.0 * train_correct / n_train

                # Test (vectorized)
                logits = model.forward(test_boards)
                masks_t = (test_boards == 0)
                logits_masked = logits.copy()
                logits_masked[~masks_t] = -np.inf
                exps = np.exp(logits_masked - np.max(logits_masked, axis=1, keepdims=True))
                probs = exps / np.sum(exps, axis=1, keepdims=True)

                target_probs = test_moves.astype(float)
                sums = np.sum(target_probs, axis=1, keepdims=True)
                target_probs = np.divide(target_probs, sums, out=np.zeros_like(target_probs), where=sums != 0)
                avg_test_loss = -np.sum(target_probs * np.log(probs + 1e-15)) / n_test

                preds = np.argmax(probs, axis=1)
                test_correct = np.sum(test_moves[np.arange(n_test), preds] == 1)
                test_acc = 100.0 * test_correct / n_test

                # Early stopping + Save best
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    epochs_no_improve = 0
                    _save_fast_model(model, best_model_path)
                    best_indicator = " [BEST]"
                else:
                    epochs_no_improve += 1
                    best_indicator = ""

                # UI update
                self.epoch_history.append(epoch + 1)
                self.loss_history.append(avg_train_loss)
                self.train_accuracy_history.append(train_acc)
                self.test_loss_history.append(avg_test_loss)
                self.test_accuracy_history.append(test_acc)

                progress = 100.0 * (epoch + 1) / epochs
                self.root.after(0, lambda p=progress: self.train_progress_var.set(p))
                self.root.after(0, self._update_metrics_display)
                prog_text = f"Epoch {epoch+1}/{epochs} | Train {train_acc:.1f}% | Test {test_acc:.1f}% | Loss {avg_train_loss:.4f}/{avg_test_loss:.4f}{best_indicator}"
                self.root.after(0, lambda t=prog_text: self.progress_text.config(text=t))

                if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                    self.root.after(0, self.update_chart)

                if epochs_no_improve >= patience:
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)", foreground="orange"))
                    break

            if self.stop_training:
                self.root.after(0, lambda: self.status_label.config(text=f"Training stopped at epoch {len(self.epoch_history)}", foreground="orange"))
            else:
                final_text = f"DONE! Train {self.train_accuracy_history[-1]:.1f}% | Test {self.test_accuracy_history[-1]:.1f}% | Best {best_test_acc:.1f}% | {os.path.basename(best_model_path)}"
                self.root.after(0, lambda t=final_text: self.status_label.config(text=t, foreground="green"))

            self.root.after(0, self.update_chart)
            self.root.after(0, self.refresh_models)
            self.current_model = load_model(best_model_path) if os.path.exists(best_model_path) else None

        except Exception as e:
            self.root.after(0, lambda e=e: self.status_label.config(text=f"ERROR: {str(e)}", foreground="red"))
        finally:
            self.is_training = False
            self.stop_training = False
            self.root.after(0, lambda: self.start_button.config(state="normal"))
            self.root.after(0, lambda: self.stop_button.config(state="disabled"))
    
    def refresh_datasets(self):
        """Aktualisiert die Liste der verfÃ¼gbaren Datasets"""
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
        # auch dataset-tab aktualisieren
        if hasattr(self, 'ds_combo_tab'):
            self.ds_paths_tab, self.ds_names_tab = get_available_datasets()
            if self.ds_names_tab:
                curr = self.ds_var_tab.get()
                self.ds_combo_tab['values'] = self.ds_names_tab
                if curr in self.ds_names_tab:
                    self.ds_var_tab.set(curr)
                else:
                    self.ds_var_tab.set(self.ds_names_tab[0])
            else:
                self.ds_combo_tab['values'] = []
                self.ds_var_tab.set("")
    
    def refresh_models(self):
        """Aktualisiert die Liste der verfÃ¼gbaren Modelle"""
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

    def start_generate_dataset(self):
        """Startet die Dataset-Generierung in einem separaten Thread"""
        try:
            size = int(self.new_dataset_size_tab_var.get())
            mode = self.new_dataset_mode_tab_var.get()
            name = self.dataset_name_var.get().strip()
        except ValueError:
            self.gen_status_label.config(text="UngÃ¼ltige GrÃ¶ÃŸe", foreground="red")
            return
        
        if not name:
            name = f"{mode}_{size}"
        
        # Flags fÃ¼r Pause/Abort
        self.gen_stop_event = threading.Event()
        self.gen_pause_event = threading.Event()
        
        # UI-Updates
        self.gen_start_button.config(state=tk.DISABLED)
        self.gen_pause_button.config(state=tk.NORMAL)
        self.gen_abort_button.config(state=tk.NORMAL)
        self.gen_progress_var.set(0)
        self.gen_status_label.config(text="Generierung lÃ¤uft...", foreground="blue")
        
        def worker():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
                dataset_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dataset_module)
                
                # Generiere Dataset mit Callback fÃ¼r Progress
                data = self._generate_dataset_with_progress(
                    dataset_module, size, mode, 
                    self.gen_stop_event, self.gen_pause_event
                )
                
                if data is None:  # Abgebrochen
                    self.root.after(0, lambda: self.gen_status_label.config(
                        text="Generierung abgebrochen", foreground="orange"
                    ))
                    return
                
                # Speichern
                os.makedirs("datasets", exist_ok=True)
                filename = f"{name}.npy"
                filepath = os.path.join("datasets", filename)
                np.save(filepath, data)
                
                # Validierung
                stats = validate_dataset(filepath)
                if stats and stats['quality'] == "OK":
                    status_text = f"Dataset '{filename}' erfolgreich generiert ({len(data)} samples, 100% korrekt)"
                    color = "green"
                else:
                    status_text = f"FEHLER: Dataset fehlerhaft! {stats}"
                    color = "red"
                
                self.root.after(0, lambda: self.gen_status_label.config(text=status_text, foreground=color))
                self.root.after(0, self.refresh_dataset_overview)
                
            except Exception as e:
                self.root.after(0, lambda: self.gen_status_label.config(
                    text=f"Fehler: {str(e)}", foreground="red"
                ))
            finally:
                self.root.after(0, lambda: self._reset_gen_buttons())
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
    
    def _generate_dataset_with_progress(self, dataset_module, size, mode, stop_event, pause_event):
        """Generiert Dataset mit Progress-Tracking"""
        if mode == "tree":
            return self._generate_tree_with_progress(dataset_module, size, stop_event, pause_event)
        else:
            return self._generate_selfplay_with_progress(dataset_module, size, stop_event, pause_event)
    
    def _generate_tree_with_progress(self, dataset_module, size, stop_event, pause_event):
        """Generiert Tree-Mode Dataset mit Progress"""
        data = []
        invalid_count = 0
        
        start_board = np.zeros(9, dtype=int)
        states = [(start_board, 1)]
        
        while states and len(data) < size:
            if stop_event.is_set():
                return None
            
            # Pause-Logik
            while pause_event.is_set():
                if stop_event.is_set():
                    return None
                import time
                time.sleep(0.1)
            
            board, player = states.pop()
            
            if not dataset_module.is_valid_board(board):
                invalid_count += 1
                continue
            
            winner = dataset_module.check_winner(board)
            if winner != 0 or np.all(board != 0):
                continue
            
            best_move = dataset_module.find_best_move_for_player(board, player)
            
            if best_move is None or board[best_move] != 0:
                continue
            
            canonical_board = dataset_module.canonicalize_board(board, player)
            data.append(np.append(canonical_board, best_move))
            
            # Progress-Update
            progress = 100 * len(data) / size
            self.root.after(0, lambda p=progress, d=len(data): [
                self.gen_progress_var.set(p),
                self.gen_progress_label.config(text=f"{d}/{size} samples")
            ])
            
            for next_board in dataset_module.get_next_boards(board, player):
                states.append((next_board, -player))
        
        return np.array(data, dtype=int) if data else None
    
    def _generate_selfplay_with_progress(self, dataset_module, size, stop_event, pause_event):
        """Generiert Selfplay-Mode Dataset mit Progress"""
        data = []
        rng = np.random.default_rng()
        
        while len(data) < size:
            if stop_event.is_set():
                return None
            
            # Pause-Logik
            while pause_event.is_set():
                if stop_event.is_set():
                    return None
                import time
                time.sleep(0.1)
            
            board = np.zeros(9, dtype=int)
            player = 1
            
            while True:
                winner = dataset_module.check_winner(board)
                if winner != 0 or np.all(board != 0):
                    break
                
                if not dataset_module.is_valid_board(board):
                    break
                
                best_moves = dataset_module.find_all_best_moves_for_player(board, player)
                if not best_moves:
                    break
                
                move = rng.choice(best_moves)
                
                if board[move] != 0:
                    break
                
                canonical_board = dataset_module.canonicalize_board(board, player)
                data.append(np.append(canonical_board, move))
                
                # Progress-Update
                if len(data) % 10 == 0:
                    progress = 100 * len(data) / size if size > 0 else 0
                    self.root.after(0, lambda p=progress, d=len(data): [
                        self.gen_progress_var.set(p),
                        self.gen_progress_label.config(text=f"{d}/{size} samples")
                    ])
                
                if len(data) >= size:
                    break
                
                board = board.copy()
                board[move] = player
                player = -player
        
        return np.array(data, dtype=int) if data else None
    
    def pause_generate_dataset(self):
        """Pausiert die Dataset-Generierung"""
        if hasattr(self, 'gen_pause_event') and not self.gen_pause_event.is_set():
            self.gen_pause_event.set()
            self.gen_pause_button.config(text="Fortsetzen")
            self.gen_status_label.config(text="Pausiert", foreground="orange")
        elif hasattr(self, 'gen_pause_event'):
            self.gen_pause_event.clear()
            self.gen_pause_button.config(text="Pausieren")
            self.gen_status_label.config(text="Generierung lÃ¤uft...", foreground="blue")
    
    def abort_generate_dataset(self):
        """Bricht die Dataset-Generierung ab"""
        if hasattr(self, 'gen_stop_event'):
            self.gen_stop_event.set()
            self.gen_status_label.config(text="Abbruch...", foreground="orange")
    
    def _reset_gen_buttons(self):
        """Setzt Buttons nach Generierung zurÃ¼ck"""
        self.gen_start_button.config(state=tk.NORMAL)
        self.gen_pause_button.config(state=tk.DISABLED, text="Pausieren")
        self.gen_abort_button.config(state=tk.DISABLED)
    
    def refresh_dataset_overview(self):
        """Aktualisiert die Dataset-Ãœbersicht mit Statistiken"""
        # LÃ¶sche alte EintrÃ¤ge
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        
        datasets, names = get_available_datasets()
        
        for path, name in zip(datasets, names):
            try:
                data = np.load(path)
                size_bytes = os.path.getsize(path)
                num_samples = len(data)
                
                # Bestimme Modus (Tree vs Selfplay vs Optimal) basierend auf Dateinamen
                if "optimal" in name.lower():
                    mode = "Optimal"
                elif "selfplay" in name.lower():
                    mode = "Selfplay"
                elif "tree" in name.lower():
                    mode = "Tree"
                else:
                    mode = "Unknown"
                
                # Validierung
                stats = validate_dataset(path)
                if stats and stats['quality'] == "OK":
                    status = "âœ“ OK"
                    tags = ("ok",)
                else:
                    status = "âœ— FEHLER"
                    tags = ("error",)
                
                # Formatiere GrÃ¶ÃŸe
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024*1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                else:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                
                self.dataset_tree.insert("", "end", text="", values=(
                    name, size_str, mode, f"{num_samples:,}", status
                ), tags=tags)
            except Exception as e:
                self.dataset_tree.insert("", "end", text="", values=(
                    name, "?", "?", "?", f"Fehler: {str(e)}"
                ), tags=("error",))
        
        # Tags fÃ¼r Farben
        self.dataset_tree.tag_configure("ok", foreground="green")
        self.dataset_tree.tag_configure("error", foreground="red")
    
    def generate_dataset(self):
        """LÃ¤sst das Minimax-Skript laufen und speichert ein neues Dataset."""
        # wÃ¤hlen, ob wir von Trainings- oder Dataset-Tab aufgerufen wurden
        if hasattr(self, 'new_dataset_size_tab_var'):
            size_var = self.new_dataset_size_tab_var
            mode_var = self.new_dataset_mode_tab_var
        else:
            size_var = self.new_dataset_size_var
            mode_var = self.new_dataset_mode_var
        try:
            size = int(size_var.get())
            mode = mode_var.get()
        except ValueError:
            if hasattr(self, 'dataset_status_label'):
                self.dataset_status_label.config(text="UngÃ¼ltige GrÃ¶ÃŸe fÃ¼r Dataset", foreground="red")
            self.status_label.config(text="UngÃ¼ltige GrÃ¶ÃŸe fÃ¼r Dataset", foreground="red")
            return
        # status anzeigen
        if hasattr(self, 'dataset_status_label'):
            self.dataset_status_label.config(text=f"Erzeuge Dataset ({mode}) ...", foreground="blue")
        self.status_label.config(text=f"Erzeuge Dataset ({mode}) ...", foreground="blue")

        def worker():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("dataset", "NeuralNetwork/training-dataset/dataset.py")
                dataset_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dataset_module)
                data = dataset_module.generate_dataset(max_boards=size, mode=mode)
                # speichern
                os.makedirs("datasets", exist_ok=True)
                filename = f"tictactoe_dataset_{len(data)}.npy"
                np.save(os.path.join("datasets", filename), data)
                if hasattr(self, 'dataset_status_label'):
                    self.root.after(0, self.dataset_status_label.config, {"text": f"Dataset fertig: {filename}", "foreground": "green"})

                self.root.after(0, self.status_label.config, {"text": f"Dataset fertig: {filename}", "foreground": "green"})
                self.root.after(0, self.refresh_datasets)
            except Exception as e:
                if hasattr(self, 'dataset_status_label'):
                    self.root.after(0, self.dataset_status_label.config, {"text": f"Fehler beim Dataset: {e}", "foreground": "red"})
                self.root.after(0, self.status_label.config, {"text": f"Fehler beim Dataset: {e}", "foreground": "red"})
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
    
    def save_current_model(self):
        """Speichert das aktuell trainierte Modell im models-Ordner"""
        if self.current_model is None:
            self.status_label.config(text="Kein Modell zum Speichern verfÃ¼gbar", foreground="red")
            return
        
        try:
            os.makedirs("models", exist_ok=True)
            
            # Try to get custom name from save_name_var
            name = None
            if hasattr(self, 'save_name_var'):
                name = self.save_name_var.get().strip()
            
            if name:
                # ensure extension and sanitize
                if not name.endswith('.npz'):
                    name = name + '.npz'
                # basic name check: no path separators
                name = os.path.basename(name)
                filepath = os.path.join("models", name)
            else:
                # Use timestamp for automatic naming
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                filepath = os.path.join("models", f"model_{timestamp}.npz")
            
            # Use the model's built-in save method
            if hasattr(self.current_model, 'save_model'):
                # Call the NeuralNetwork's save_model method properly
                # Save to directory, then move if needed
                weights = {}
                for i, layer in enumerate(self.current_model.layers):
                    for j, neuron in enumerate(layer.neurons):
                        weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                        weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
                
                np.savez(filepath, **weights)
                self.status_label.config(text=f"âœ“ Modell gespeichert: {os.path.basename(filepath)}", foreground="green")
            else:
                self.status_label.config(text="Fehler: Modell hat keine save_model Methode", foreground="red")
                return
            
            self.refresh_models()
        except Exception as e:
            import traceback
            error_msg = f"Fehler beim Speichern: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            print(f"Save error details: {traceback.format_exc()}")
    
    def test_model(self):
        """Testet ein geladenes Modell auf beiden Test-Methoden (realistisch + zufÃ¤llig)"""
        if self.is_training:
            return
        
        
        selected_name = self.model_var.get()
        if not selected_name:
            self.status_label.config(text="Kein Modell ausgewÃ¤hlt", foreground="red")
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
                model = load_model(model_path)
                if model is None:
                    self.root.after(0, self.status_label.config, {"text": "Fehler: Modell konnte nicht geladen werden", "foreground": "red"})
                    return
                # propagate architecture back to UI
                num_hidden = max(len(model.layers) - 1, 0)
                hidden = len(model.layers[0].neurons) if num_hidden > 0 else 0
                self.root.after(0, self.num_layers_var.set, str(num_hidden))
                self.root.after(0, self.hidden_neurons_var.set, str(hidden))

                # Test 1: Realistische SpielverlÃ¤ufe
                self.root.after(0, self.status_label.config, {"text": "Teste Modell auf realistischen SpielverlÃ¤ufen...", "foreground": "blue"})
                realistic_accuracy, realistic_total, realistic_correct = test_on_realistic_games(model, num_test_boards=100)
                
                # Test 2: ZufÃ¤llige Boards
                self.root.after(0, self.status_label.config, {"text": "Teste Modell auf zufÃ¤lligen SpielstÃ¤nden...", "foreground": "blue"})
                random_accuracy, random_total, random_correct = test_model_on_random_boards(model, num_test_boards=100)
                
                # Beide Ergebnisse anzeigen
                result_text = (f"Test abgeschlossen!\n"
                              f"Realistische Spiele: {realistic_correct}/{realistic_total} korrekt ({realistic_accuracy:.2f}%)\n"
                              f"ZufÃ¤llige Boards: {random_correct}/{random_total} korrekt ({random_accuracy:.2f}%)")
                
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

    def start_tests(self):
        """Startet die ausgewÃ¤hlten Tests in einem Worker-Thread"""
        if self.test_running:
            return
        
        selected_name = self.test_model_var.get()
        if not selected_name:
            self.test_status_label.config(text="Kein Modell ausgewÃ¤hlt", foreground="red")
            return
        
        try:
            num_games = int(self.test_num_games_var.get())
            if num_games < 1:
                raise ValueError("Anzahl Spiele muss >= 1 sein")
        except ValueError as e:
            self.test_status_label.config(text=f"Fehler: {str(e)}", foreground="red")
            return
        
        # Modell-Pfad finden
        model_path = None
        if selected_name and self.test_model_paths:
            try:
                idx = self.test_model_names.index(selected_name)
                model_path = self.test_model_paths[idx]
            except (ValueError, IndexError):
                self.test_status_label.config(text="Fehler: Modell nicht gefunden", foreground="red")
                return
        
        test_type = self.test_type_var.get()
        
        self.test_running = True
        self.test_start_button.config(state="disabled")
        self.test_results_text.delete(1.0, tk.END)
        
        def test_worker():
            try:
                model = load_model(model_path)
                if model is None:
                    self.root.after(0, self.test_status_label.config, 
                                   {"text": "Fehler: Modell konnte nicht geladen werden", "foreground": "red"})
                    return
                
                # Importiere game_test Module
                import importlib.util
                spec = importlib.util.spec_from_file_location("game_test", "game_test.py")
                game_test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(game_test_module)
                
                result_lines = []
                result_lines.append(f"Test-Typ: {test_type}\n")
                result_lines.append(f"Modell: {selected_name}\n")
                result_lines.append(f"Anzahl Spiele: {num_games}\n")
                result_lines.append("=" * 60 + "\n\n")
                
                if test_type == "Model vs Minimax":
                    self.root.after(0, self.test_status_label.config, 
                                   {"text": "FÃ¼hre Spiele durch: Model vs Minimax", "foreground": "blue"})
                    
                    # Create player functions
                    nn_func = game_test_module.nn_player(model)
                    minimax_func = game_test_module.minimax_player
                    
                    # Run games
                    p1_wins, draws, p2_wins = game_test_module.run_games(
                        "Modell", nn_func,
                        "Minimax", minimax_func,
                        num_games
                    )
                    
                    result_lines.append(f"Model Wins: {p1_wins}/{num_games} ({p1_wins/num_games*100:.1f}%)\n")
                    result_lines.append(f"Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)\n")
                    result_lines.append(f"Minimax Wins: {p2_wins}/{num_games} ({p2_wins/num_games*100:.1f}%)\n")
                    
                elif test_type == "Model vs Random":
                    self.root.after(0, self.test_status_label.config, 
                                   {"text": "FÃ¼hre Spiele durch: Model vs Random", "foreground": "blue"})
                    
                    # Create player functions
                    nn_func = game_test_module.nn_player(model)
                    random_func = game_test_module.random_player
                    
                    # Run games
                    p1_wins, draws, p2_wins = game_test_module.run_games(
                        "Modell", nn_func,
                        "Zufallsspiel", random_func,
                        num_games
                    )
                    
                    result_lines.append(f"Model Wins: {p1_wins}/{num_games} ({p1_wins/num_games*100:.1f}%)\n")
                    result_lines.append(f"Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)\n")
                    result_lines.append(f"Random Wins: {p2_wins}/{num_games} ({p2_wins/num_games*100:.1f}%)\n")
                    
                elif test_type == "Fair Accuracy":
                    self.root.after(0, self.test_status_label.config, 
                                   {"text": "Berechne Fair Accuracy...", "foreground": "blue"})
                    
                    # Import evaluate_fair_accuracy
                    spec2 = importlib.util.spec_from_file_location("evaluate_fair_accuracy", "evaluate_fair_accuracy.py")
                    eval_module = importlib.util.module_from_spec(spec2)
                    spec2.loader.exec_module(eval_module)
                    
                    # Get latest dataset
                    datasets, _ = get_available_datasets()
                    if not datasets:
                        result_lines.append("Fehler: Kein Dataset gefunden!\n")
                    else:
                        for dataset_path in datasets[:3]:  # Test auf neueste 3 Datasets
                            dataset_name = os.path.basename(dataset_path)
                            accuracy, total, correct = eval_module.compute_fair_accuracy(
                                model, dataset_path, min(500, num_games)
                            )
                            result_lines.append(f"Dataset: {dataset_name}\n")
                            result_lines.append(f"Fair Accuracy: {accuracy:.2f}% ({correct}/{total})\n\n")
                
                # Update UI
                result_text = "".join(result_lines)
                self.root.after(0, self.test_results_text.insert, tk.END, result_text)
                self.root.after(0, self.test_status_label.config, 
                               {"text": "Tests abgeschlossen!", "foreground": "green"})
                self.root.after(0, self.test_progress_var.set, 100)
                
            except Exception as e:
                import traceback
                error_msg = f"Fehler beim Testen: {str(e)}\n{traceback.format_exc()}"
                self.root.after(0, self.test_results_text.insert, tk.END, error_msg)
                self.root.after(0, self.test_status_label.config, 
                               {"text": f"Fehler: {str(e)}", "foreground": "red"})
            finally:
                self.root.after(0, self.test_start_button.config, {"state": "normal"})
                self.root.after(0, lambda: setattr(self, 'test_running', False))
        
        thread = threading.Thread(target=test_worker)
        thread.daemon = True
        thread.start()
    
    def _refresh_test_models(self):
        """Auto-refresh models in tests tab dropdown"""
        self.test_model_paths, new_names = get_available_models()
        self.test_model_combo['values'] = new_names
        if new_names and self.test_model_var.get() not in new_names:
            self.test_model_var.set(new_names[0])
    
    def _refresh_play_models(self):
        """Auto-refresh models in play tab dropdown"""
        self.play_model_paths, new_names = get_available_models()
        self.play_model_combo['values'] = new_names
        if new_names and self.play_model_var.get() not in new_names:
            self.play_model_var.set(new_names[0])
    
    def _update_metrics_display(self):
        """Update the metrics display panel during training"""
        if not hasattr(self, 'metrics_display'):
            return
        
        self.metrics_display.config(state=tk.NORMAL)
        self.metrics_display.delete(1.0, tk.END)
        
        # Build metrics text
        metrics_text = "Epoche: -\n"
        
        if len(self.epoch_history) > 0:
            epoch = int(self.epoch_history[-1])
            metrics_text = f"Epoche: {epoch}\n"
        
        metrics_text += "\nTrain:\n"
        if len(self.loss_history) > 0:
            metrics_text += f"  Loss:  {self.loss_history[-1]:.4f}\n"
        if len(self.train_accuracy_history) > 0:
            metrics_text += f"  Acc:   {self.train_accuracy_history[-1]:.1f}%\n"
        
        metrics_text += "\nTest:\n"
        if len(self.test_loss_history) > 0:
            metrics_text += f"  Loss:  {self.test_loss_history[-1]:.4f}\n"
        if len(self.test_accuracy_history) > 0:
            metrics_text += f"  Acc:   {self.test_accuracy_history[-1]:.1f}%\n"
        
        metrics_text += "\nBest:\n"
        if len(self.test_accuracy_history) > 0:
            best_acc = max(self.test_accuracy_history)
            metrics_text += f"  Acc:   {best_acc:.1f}%\n"
        
        self.metrics_display.insert(1.0, metrics_text)
        self.metrics_display.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()
