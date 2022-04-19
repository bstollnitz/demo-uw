"""Main code."""

import numpy as np
import pandas as pd

XOR_table = pd.read_csv('demo-uw/data/XOR_table.csv')

XOR_table = XOR_table.values
X = XOR_table[:, :2]
targets = XOR_table[:, -1].reshape(-1, 1)

print(X)  # Input data
print(targets)  # Output targets

input_dim, hidden_dim, output_dim = 2, 16, 1
learning_rate = 0.1

# Define a hidden layer
W1 = np.random.randn(input_dim, hidden_dim)
# Define an output layer
W2 = np.random.randn(hidden_dim, output_dim)
# Define biases
b1 = np.random.randn(1, hidden_dim)
b2 = np.random.randn(1, output_dim)


# Define activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


Loss = []

# Training
for i in range(10000):
    # Forward pass: compute predicted y (Activation function of choice)
    z = sigmoid(np.dot(X, W1) + b1)
    y = sigmoid(np.dot(z, W2) + b2)

    # Compute and print loss (L2 norm loss)
    loss = 1 / 4 * np.sum((y - targets)**2)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_W2 = 2 * (np.dot(z.T, (y - targets) * y * (1 - y)))
    grad_W1 = 2 * np.dot(
        X.T,
        np.dot((y - targets) * y * (1 - y), W2.T) * z * (1 - z))
    grad_b2 = np.sum(2 * (y - targets) * y * (1 - y), axis=0, keepdims=True)
    grad_b1 = np.sum(2 * np.dot(
        (y - targets) * y * (1 - y), W2.T) * z * (1 - z),
                     axis=0,
                     keepdims=True)

    # Update weights
    W2 = W2 - learning_rate * grad_W2
    W1 = W1 - learning_rate * grad_W1
    b2 = b2 - learning_rate * grad_b2
    b1 = b1 - learning_rate * grad_b1

    # Save loss to an array
    Loss.append(loss)
