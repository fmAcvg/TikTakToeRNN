import numpy as np

def combine_datasets():
    """
    Combine the original dataset with the critical situations dataset.
    """
    original_data = np.load('datasets/tictactoe_dataset_augmented.npy')
    critical_data = np.load('datasets/tictactoe_critical.npy')

    combined_data = np.vstack([original_data, critical_data])
    np.save('datasets/tictactoe_combined.npy', combined_data)
    print(f"Combined dataset shape: {combined_data.shape}")

if __name__ == "__main__":
    combine_datasets()