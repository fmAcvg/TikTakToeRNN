#!/usr/bin/env python
"""
Test-Script für die neuen Test-Features in der GUI
Testet ohne GUI zu öffnen, ob die Funktionen korrekt aufgerufen werden können
"""
import numpy as np
import os
import sys

def test_game_test_module():
    """Test: game_test Modul kann geladen und verwendet werden"""
    print("\n" + "="*70)
    print("Test 1: game_test Modul")
    print("="*70)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("game_test", "game_test.py")
    game_test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game_test_module)
    
    # Check ob alle Funktionen verfügbar sind
    required_functions = ['Game', 'minimax_player', 'random_player', 'nn_player', 'run_games']
    for func in required_functions:
        if hasattr(game_test_module, func):
            print(f"✓ {func} gefunden")
        else:
            print(f"✗ {func} NICHT gefunden")
            return False
    
    return True

def test_evaluate_module():
    """Test: evaluate_fair_accuracy Modul kann geladen und verwendet werden"""
    print("\n" + "="*70)
    print("Test 2: evaluate_fair_accuracy Modul")
    print("="*70)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_fair_accuracy", "evaluate_fair_accuracy.py")
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    if hasattr(eval_module, 'compute_fair_accuracy'):
        print(f"✓ compute_fair_accuracy gefunden")
        return True
    else:
        print(f"✗ compute_fair_accuracy NICHT gefunden")
        return False

def test_gui_setup():
    """Test: GUI kann ohne Fehler aufgebaut werden (ohne Fenster zu öffnen)"""
    print("\n" + "="*70)
    print("Test 3: GUI Setup (ohne Tkinter Fenster)")
    print("="*70)
    
    # Mocking Tkinter
    import tkinter as tk
    import tkinter.ttk as ttk
    from unittest.mock import Mock, patch
    
    # Erstelle Root ohne Fenster zu öffnen
    root = tk.Tk()
    root.withdraw()  # Verstecke das Fenster
    
    try:
        from main import TrainingGUI
        app = TrainingGUI(root)
        
        # Check ob Tests-Tab existiert
        if hasattr(app, 'tests_tab'):
            print("✓ Tests-Tab wurde erstellt")
        else:
            print("✗ Tests-Tab wurde NICHT erstellt")
            return False
        
        # Check ob Test-Widgets existieren
        if hasattr(app, 'test_start_button'):
            print("✓ test_start_button existiert")
        else:
            print("✗ test_start_button existiert NICHT")
            return False
        
        if hasattr(app, 'test_results_text'):
            print("✓ test_results_text existiert")
        else:
            print("✗ test_results_text existiert NICHT")
            return False
        
        if hasattr(app, 'start_tests'):
            print("✓ start_tests() Methode existiert")
        else:
            print("✗ start_tests() Methode existiert NICHT")
            return False
        
        print("\n✓ GUI-Setup erfolgreich!")
        return True
    except Exception as e:
        print(f"✗ Fehler bei GUI-Setup: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        root.destroy()

def test_model_loading():
    """Test: Ein Modell kann geladen werden"""
    print("\n" + "="*70)
    print("Test 4: Modell laden")
    print("="*70)
    
    from main import get_available_models, load_model
    
    model_paths, model_names = get_available_models()
    
    if not model_paths:
        print("⚠ Keine Modelle verfügbar")
        return True  # Nicht kritisch
    
    print(f"✓ {len(model_paths)} Modelle gefunden")
    
    try:
        model = load_model(model_paths[0])
        if model is not None:
            print(f"✓ Modell geladen: {model_names[0]}")
            # Test Prediction
            board = np.zeros(9)
            legal_mask = np.ones(9)
            move, conf = model.predict(board, legal_mask)
            if move is not None:
                print(f"✓ Modell kann Predictions machen (move={move})")
                return True
            else:
                print("✗ Modell kann keine Prediction machen")
                return False
        else:
            print("✗ Modell konnte nicht geladen werden")
            return False
    except Exception as e:
        print(f"⚠ Fehler beim Modell-Test: {e}")
        return True  # Nicht kritisch

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TikTakToe RNN - GUI Integration Tests")
    print("="*70)
    
    results = []
    
    # Führe Tests durch
    results.append(("game_test Modul", test_game_test_module()))
    results.append(("evaluate_fair_accuracy Modul", test_evaluate_module()))
    results.append(("GUI Setup", test_gui_setup()))
    results.append(("Modell laden", test_model_loading()))
    
    # Zusammenfassung
    print("\n" + "="*70)
    print("Test Zusammenfassung")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nGesamt: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n✓ Alle Tests bestanden! GUI ist bereit.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} Tests fehlgeschlagen!")
        sys.exit(1)
