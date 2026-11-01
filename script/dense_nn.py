# TODO: loads weights and dataset, implement the model and report accuracy


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mnist_data(csv_path):
    """Load MNIST dataset from CSV file."""
    print('Reading CSV Data...')
    data = pd.read_csv(csv_path)
    return data

def preprocess_data(data):
    """Preprocess MNIST data - normalize features and separate labels."""
    # Separate labels (first column) from features
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:].values

    # Normalize features to [0, 1]
    features = features / 255.0

    return features, labels

def load_weights_and_biases(weight_dir):
    """Load pre-trained weights and biases."""
    W1 = pd.read_csv(f'{weight_dir}/weights_hidden.csv', header=None).values
    W2 = pd.read_csv(f'{weight_dir}/weights_output.csv', header=None).values
    B1 = pd.read_csv(f'{weight_dir}/biases_hidden.csv', header=None).values.flatten()
    B2 = pd.read_csv(f'{weight_dir}/biases_output.csv', header=None).values.flatten()
    return W1, W2, B1, B2

def forward_pass(X, W1, W2, B1, B2):
    """
    Forward pass of the dense neural network.
    H = tanh(X * W1^T + B1)
    Z = sigmoid(H * W2^T + B2)
    Y = argmax(Z)
    """

    # TODO: IMPLEMENT THIS
    batch = X.shape[0]
    Z = np.zeros((batch, W2.shape[0]))
    Y = np.zeros(batch, dtype=int)
    return Y, Z

def calculate_accuracy(predictions, labels):
    """Calculate prediction accuracy."""
    return np.mean(predictions == labels)

def main():
    """Main function to run dense NN."""
    # TODO: Load weights and dataset, implement the model and report accuracy

    print('Loading data...')
    data = load_mnist_data('./data/mnist_train.csv')
    features, labels = preprocess_data(data)

    print(f'Data shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')

    print('Loading weights and biases...')
    W1, W2, B1, B2 = load_weights_and_biases('./data/model')

    print(f'W1 shape: {W1.shape}')
    print(f'W2 shape: {W2.shape}')
    print(f'B1 shape: {B1.shape}')
    print(f'B2 shape: {B2.shape}')

    print('Running forward propagation...')
    predictions, outputs = forward_pass(features, W1, W2, B1, B2)

    accuracy = calculate_accuracy(predictions, labels)
    print(f'Dense Neural Network Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
