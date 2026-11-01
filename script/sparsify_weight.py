import sys
import csv
import numpy as np
import pandas as pd

FOLDER = './data/model/'

def magnitude_based_pruning(weights, sparsity):
    """
    Apply magnitude-based pruning to weight matrix.

    Args:
        weights: Input weight matrix
        sparsity: Sparsity level (0-1), e.g., 0.8 means keep 20% of weights

    Returns:
        Pruned weight matrix
    """
    pruned = weights.copy()
    # TODO : Implement magnitude-based pruning
    return pruned

def load_weights(filename):
    """Load weights from CSV file."""
    return pd.read_csv(filename, header=None).values

def save_weights(weights, filename):
    """Save weights to CSV file."""
    pd.DataFrame(weights).to_csv(filename, header=False, index=False)

def main():
    """
    Load weights and sparsify them using magnitude-based pruning.
    Save pruned weights with naming convention: <sparsity_level>_W1.csv, <sparsity_level>_W2.csv
    """
    print("Loading weights...")
    W1 = load_weights(f'{FOLDER}weights_hidden.csv')
    W2 = load_weights(f'{FOLDER}weights_output.csv')

    print(f'W1 shape: {W1.shape}')
    print(f'W2 shape: {W2.shape}')

    # Sparsity levels from 50% to 95% in steps of 5%
    sparsity_levels = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    print("Pruning weights for different sparsity levels...")
    for sparsity in sparsity_levels:
        sparsity_ratio = sparsity / 100.0

        print(f'  Processing sparsity level: {sparsity}%')

        # Prune weights
        W1_pruned = magnitude_based_pruning(W1, sparsity_ratio)
        W2_pruned = magnitude_based_pruning(W2, sparsity_ratio)

        # Save pruned weights with naming convention: <sparsity>_W1.csv, <sparsity>_W2.csv
        save_weights(W1_pruned, f'{FOLDER}{sparsity}_W1.csv')
        save_weights(W2_pruned, f'{FOLDER}{sparsity}_W2.csv')

        # Calculate actual sparsity achieved
        actual_sparsity_w1 = np.sum(W1_pruned == 0) / W1_pruned.size * 100
        actual_sparsity_w2 = np.sum(W2_pruned == 0) / W2_pruned.size * 100

        print(f'    W1 actual sparsity: {actual_sparsity_w1:.2f}%')
        print(f'    W2 actual sparsity: {actual_sparsity_w2:.2f}%')

    print("Sparsification complete!")

if __name__ == '__main__':
    main()
