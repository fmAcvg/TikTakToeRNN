# TicTacToe RNN

Trainiere ein neuronales Netzwerk um TicTacToe zu spielen.

## Starten

python main.py

## CLI-Training (train_improved.py)

Du kannst `train_improved.py` jetzt direkt mit Hyperparametern und Modellnamen starten.

Beispiel:

python train_improved.py --hidden-layers 6 --neurons 384 --lr 0.002 --batch-size 64 --epochs 400 --model-name mein_bestes_modell

Wichtige Parameter:

- `--hidden-layers` Anzahl Hidden-Layer
- `--neurons` Neuronen pro Hidden-Layer
- `--lr` Learning Rate
- `--weight-decay` L2-Regularisierung
- `--batch-size` Batch-Größe
- `--epochs` Maximale Epochen
- `--patience` Early-Stopping-Patience
- `--test-split` Testsplit (z. B. `0.2`)
- `--multi-hot` / `--single-hot`
- `--split-unique` / `--split-sample`
- `--same-train-test` (Debug-Modus)
- `--dataset-path` Pfad zu `.npy` Dataset
- `--model-name` Benutzerdefinierter Name für das gespeicherte Modell

Alle Optionen anzeigen:

python train_improved.py --help

## Workflow (4 Tabs)

Das Tool führt dich durch die komplette Pipeline:

1. **Training** - Konfiguriere und trainiere das Modell
2. **Dataset** - Generiere oder lade Trainingsdaten  
3. **Testing** - Teste die Modellgenauigkeit
4. **Spielen** - Spielen gegen das trainierte Netzwerk

Jeder Tab baut auf den Ergebnissen des vorherigen auf.

## Architektur

- Input: 9 Positionen (33 Board)
- Hidden: Konfigurierbar (default 128, tanh Aktivation)
- Output: 9 Züge (softmax)

## Training

- Dataset: Minimax-generierte optimale Züge
- Loss: Cross-Entropy 
- Optimierer: SGD
- Early Stopping: Verhindert Overfitting

## Modelle

Trainierte Modelle werden im /models Ordner gespeichert (.npz Format).

## Datasets  

Trainingsdaten im /datasets Ordner.
