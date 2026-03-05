#!/usr/bin/env python3
"""
Test-Script für die integrierte Pipeline GUI
Überprüft alle Dependencies und Funktionen
"""

import sys
import os
import importlib

def check_module(module_name, display_name):
    """Überprüfe ob ein Modul verfügbar ist"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False

def check_file(path, description):
    """Überprüfe ob eine Datei existiert"""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path}")
        return False

def check_directory(path, description):
    """Überprüfe ob ein Ordner existiert"""
    if os.path.isdir(path):
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(path) for f in filenames)
        size_mb = size / (1024*1024)
        print(f"✓ {description}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ {description}: {path}")
        return False

def main():
    print("="*70)
    print("TicTacToe RNN Pipeline GUI - System Check")
    print("="*70)
    print()
    
    # Python Version
    print("Python Version:")
    print(f"  {sys.version}")
    if sys.version_info >= (3, 8):
        print("  ✓ Version OK (3.8+)")
    else:
        print("  ✗ Version zu alt! Brauchst Python 3.8+")
    print()
    
    # Required Modules
    print("Required Modules:")
    modules_ok = all([
        check_module("tkinter", "tkinter (GUI Framework)"),
        check_module("numpy", "numpy (Numerics)"),
        check_module("threading", "threading (Async)"),
    ])
    print()
    
    # Project Files
    print("Project Structure:")
    files_ok = all([
        check_file("launch_pipeline_gui.py", "Launcher Script"),
        check_file("game/integrated_pipeline_gui.py", "Pipeline GUI"),
        check_file("game/tiktaktoe.py", "Game Logic"),
    ])
    print()
    
    # Project Directories
    print("Data Directories:")
    dirs_ok = all([
        check_directory("NeuralNetwork", "Neural Network Module"),
        check_directory("datasets", "Datasets Folder"),
        check_directory("models", "Models Folder"),
    ])
    print()
    
    # Neural Network modules
    print("Neural Network Modules:")
    nn_ok = all([
        check_file("NeuralNetwork/predict.py", "Prediction Module"),
        check_file("NeuralNetwork/training.py", "Training Module"),
    ])
    print()
    
    # Summary
    print("="*70)
    if modules_ok and files_ok and dirs_ok and nn_ok:
        print("✓ All checks passed! Ready to go!")
        print()
        print("Start the GUI with:")
        print("  python launch_pipeline_gui.py")
        print()
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
