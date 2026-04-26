# ML_Practical/exp6_perceptron.py
# Experiment 6: To implementation of AND Gate Classification using Single Layer Feed Forward Perceptron Network Model in Python

import numpy as np

# Binary inputs and targets for AND Gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Initialization
W = np.zeros(2)
b = 0
lr = 0.1
epochs = 10

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        net_input = np.dot(X[i], W) + b
        y_pred = 1 if net_input > 0 else 0
        error = Y[i] - y_pred
        
        # Weight update
        W += lr * error * X[i]
        b += lr * error

# Output
print("--- Single Layer Perceptron (AND Gate) ---")
print("Trained Weights:", W)
print("Trained Bias:", round(b, 4))
print("\nPredictions:")
for i in range(len(X)):
    pred = 1 if (np.dot(X[i], W) + b) > 0 else 0
    print(f"Input: {X[i]} -> Prediction: {pred}")