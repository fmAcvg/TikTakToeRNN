#!/usr/bin/env python3
"""
Simplified TicTacToe GUI - Focus on what matters
- Clean, simple interface
- 50 epochs default
- 32 neurons (300-400 params)
- Fast, reliable training
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import threading
from datetime import datetime
import sys

sys.path.insert(0, '.')

from NeuralNetwork.predict import NeuralNetwork, create_layer
from NeuralNetwork.training import train_on_data_set
from main import get_available_datasets, load_model


class SimpleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TicTacToe RNN - Simple Training")
        self.root.geometry("900x700")
        
        self.model = None
        self.training = False
        self.board_state = np.zeros(9, dtype=int)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup simple, focused UI"""
        
        # Top: Title
        title = ttk.Label(self.root, text="TicTacToe Neural Network", 
                         font=("Helvetica", 18, "bold"))
        title.pack(pady=10)
        
        # === NOTEBOOK ===
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # TAB 1: TRAINING
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="🎓 Train")
        self.setup_train_tab(train_tab)
        
        # TAB 2: TEST
        test_tab = ttk.Frame(notebook)
        notebook.add(test_tab, text="📈 Test")
        self.setup_test_tab(test_tab)
        
        # TAB 3: PLAY
        play_tab = ttk.Frame(notebook)
        notebook.add(play_tab, text="🎮 Play")
        self.setup_play_tab(play_tab)
    
    def setup_train_tab(self, parent):
        """Simple training tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Config
        config_frame = ttk.LabelFrame(frame, text="Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(config_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(config_frame, text="(Default: 50)").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="0.005")
        ttk.Entry(config_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(config_frame, text="(Default: 0.005)").grid(row=1, column=2, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=tk.EW)
        
        self.train_btn = ttk.Button(btn_frame, text="▶️ START TRAINING", command=self.start_train)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ STOP", command=self.stop_train, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="💾 SAVE MODEL", command=self.save_model).pack(side=tk.LEFT, padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(progress_frame, height=15, width=100)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)
    
    def setup_test_tab(self, parent):
        """Simple test tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(frame, text="▶️ RUN TESTS", command=self.run_tests).pack(pady=10)
        
        self.test_text = tk.Text(frame, height=20, width=100)
        self.test_text.pack(fill=tk.BOTH, expand=True)
        self.test_text.config(state=tk.DISABLED)
    
    def setup_play_tab(self, parent):
        """Simple play tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Game board
        board_frame = ttk.LabelFrame(frame, text="Board (You=X, AI=O)", padding="10")
        board_frame.pack(pady=10)
        
        self.game_buttons = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                btn = tk.Button(board_frame, text="", width=10, height=5,
                               font=("Arial", 20),
                               command=lambda r=r, c=c: self.player_move(r, c))
                btn.grid(row=r, column=c, padx=2, pady=2)
                self.game_buttons[r][c] = btn
        
        # Info
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.game_status = ttk.Label(info_frame, text="Load a model first", 
                                    font=("Arial", 12, "bold"))
        self.game_status.pack()
        
        ttk.Button(info_frame, text="NEW GAME", command=self.new_game).pack(pady=5)
    
    def log(self, message):
        """Log to status text"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.update()
        self.status_text.config(state=tk.DISABLED)
    
    def start_train(self):
        """Start training"""
        if self.training:
            return
        
        try:
            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            self.log("❌ Invalid config")
            return
        
        self.training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._train_worker, args=(epochs, lr))
        thread.daemon = True
        thread.start()
    
    def _train_worker(self, epochs, lr):
        """Training worker thread"""
        try:
            self.log("=" * 70)
            self.log("TicTacToe RNN Training - Simple")
            self.log("=" * 70)
            self.log(f"\nConfig: Epochs={epochs}, LR={lr}")
            self.log(f"Model: 9 → 32 → 9 (~400 params)\n")
            
            # Load dataset
            datasets, names = get_available_datasets()
            if not datasets:
                self.log("❌ No datasets found")
                return
            
            data = np.load(datasets[0])
            boards = data[:, :9].astype(float)
            labels = data[:, 9:].astype(float)
            
            self.log(f"✓ Loaded {len(boards)} samples\n")
            
            # Split
            split = int(len(boards) * 0.8)
            idx = np.random.permutation(len(boards))
            
            train_b, train_l = boards[idx[:split]], labels[idx[:split]]
            test_b, test_l = boards[idx[split:]], labels[idx[split:]]
            
            # Model
            model = NeuralNetwork([
                create_layer(9, 32),
                create_layer(32, 9)
            ])
            
            self.log(f"✓ Model created: 9 → 32 → 9")
            self.log(f"✓ Train: {len(train_b)}, Test: {len(test_b)}\n")
            self.log("Training...\n")
            
            best_acc = 0
            no_improve = 0
            
            for epoch in range(epochs):
                if not self.training:
                    break
                
                # Train
                train_loss, train_correct = 0, 0
                for idx in np.random.permutation(len(train_b)):
                    loss, correct = train_on_data_set(
                        model, train_b[idx], train_l[idx], lr, 0.0, None, False
                    )
                    train_loss += loss
                    train_correct += int(correct)
                
                train_acc = 100 * train_correct / len(train_b)
                train_loss /= len(train_b)
                
                # Test
                test_loss, test_correct = 0, 0
                for idx in range(len(test_b)):
                    loss, correct = train_on_data_set(
                        model, test_b[idx], test_l[idx], 0, 0.0, None, False
                    )
                    test_loss += loss
                    test_correct += int(correct)
                
                test_acc = 100 * test_correct / len(test_b)
                test_loss /= len(test_b)
                
                # Progress
                progress = 100 * (epoch + 1) / epochs
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    no_improve = 0
                    marker = " ✓"
                else:
                    no_improve += 1
                    marker = ""
                
                if (epoch + 1) % 5 == 0:
                    self.log(f"Epoch {epoch+1:3d}/{epochs} | "
                            f"Train: {train_acc:6.2f}% | "
                            f"Test: {test_acc:6.2f}% {marker}")
                
                if no_improve >= 15:
                    self.log(f"\n⊘ Early stopping at epoch {epoch+1}")
                    break
            
            self.model = model
            self.log(f"\n✓ Training complete! Best: {best_acc:.2f}%")
            
        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            self.training = False
            self.root.after(0, lambda: (
                self.train_btn.config(state=tk.NORMAL),
                self.stop_btn.config(state=tk.DISABLED)
            ))
    
    def stop_train(self):
        self.training = False
        self.log("\n⊘ Training stopped")
    
    def save_model(self):
        if self.model is None:
            self.log("❌ No model to save")
            return
        
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = f"models/simple_{timestamp}.npz"
        
        weights = {}
        for i, layer in enumerate(self.model.layers):
            for j, neuron in enumerate(layer.neurons):
                weights[f'layer_{i}_neuron_{j}_weights'] = neuron.weights
                weights[f'layer_{i}_neuron_{j}_bias'] = neuron.bias
        
        np.savez(path, **weights)
        self.log(f"✓ Model saved: {path}")
    
    def run_tests(self):
        if self.model is None:
            self.log("❌ No model to test")
            return
        
        self.test_text.config(state=tk.NORMAL)
        self.test_text.delete(1.0, tk.END)
        self.test_text.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._test_worker)
        thread.daemon = True
        thread.start()
    
    def _test_worker(self):
        try:
            datasets, names = get_available_datasets()
            if not datasets:
                return
            
            data = np.load(datasets[0])
            boards = data[:, :9].astype(float)
            labels = data[:, 9:].astype(float)
            
            correct = 0
            loss = 0
            
            for i in range(len(boards)):
                output = self.model.forward(boards[i])
                exps = np.exp(output - np.max(output))
                probs = exps / np.sum(exps)
                
                loss += -np.sum(labels[i] * np.log(probs + 1e-15))
                if np.argmax(probs) == np.argmax(labels[i]):
                    correct += 1
            
            acc = 100 * correct / len(boards)
            loss /= len(boards)
            
            text = f"Test Results:\n{'='*40}\nAccuracy: {acc:.2f}%\nLoss: {loss:.4f}\nSamples: {len(boards)}"
            self.root.after(0, lambda t=text: (
                self.test_text.config(state=tk.NORMAL),
                self.test_text.insert(tk.END, t),
                self.test_text.config(state=tk.DISABLED)
            ))
        except Exception as e:
            self.root.after(0, lambda e=e: (
                self.test_text.config(state=tk.NORMAL),
                self.test_text.insert(tk.END, f"❌ Error: {e}"),
                self.test_text.config(state=tk.DISABLED)
            ))
    
    def new_game(self):
        if self.model is None:
            self.game_status.config(text="❌ Load a model first")
            return
        
        self.board_state = np.zeros(9, dtype=int)
        for r in range(3):
            for c in range(3):
                self.game_buttons[r][c].config(text="", fg="black", state=tk.NORMAL)
        
        self.game_status.config(text="Your turn (X)")
    
    def player_move(self, r, c):
        if self.model is None or self.board_state[r*3 + c] != 0:
            return
        
        self.board_state[r*3 + c] = 1
        self.game_buttons[r][c].config(text="X", fg="blue", state=tk.DISABLED)
        
        if self._check_win():
            self.game_status.config(text="✓ You won!")
            return
        
        # AI move
        output = self.model.forward(self.board_state.astype(float))
        moves = np.argsort(-output)
        
        for move in moves:
            if self.board_state[move] == 0:
                self.board_state[move] = -1
                r, c = move // 3, move % 3
                self.game_buttons[r][c].config(text="O", fg="red", state=tk.DISABLED)
                
                if self._check_win():
                    self.game_status.config(text="✗ AI won!")
                else:
                    self.game_status.config(text="Your turn (X)")
                return
        
        self.game_status.config(text="Draw!")
    
    def _check_win(self):
        b = self.board_state.reshape(3, 3)
        for i in range(3):
            if b[i, 0] != 0 and np.all(b[i, :] == b[i, 0]):
                return True
            if b[0, i] != 0 and np.all(b[:, i] == b[0, i]):
                return True
        if b[0, 0] != 0 and np.all(np.diag(b) == b[0, 0]):
            return True
        if b[0, 2] != 0 and np.all(np.diag(np.fliplr(b)) == b[0, 2]):
            return True
        return np.all(b != 0)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()
