import numpy as np

def combine_advanced_datasets():
    """
    Combine the original dataset with the advanced critical situations dataset.
    """
    original_data = np.load('datasets/tictactoe_dataset_augmented.npy')
    advanced_critical_data = np.load('datasets/tictactoe_advanced_critical.npy')

    combined_data = np.vstack([original_data, advanced_critical_data])
    np.save('datasets/tictactoe_combined_advanced.npy', combined_data)
    print(f"Combined dataset shape: {combined_data.shape}")

if __name__ == "__main__":
    combine_advanced_datasets()