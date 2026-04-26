# ML_Practical/exp4_hebbian.py
# Experiment 4: To implementation of AND Gate using Hebbian Learning Algorithm

import numpy as np
import matplotlib.pyplot as plt

# Bipolar representation for AND gate: [x1, x2, bias]
X = np.array([
    [-1, -1, 1], 
    [-1,  1, 1], 
    [ 1, -1, 1], 
    [ 1,  1, 1]
])
# Bipolar targets
Y = np.array([-1, -1, -1, 1])

weights = np.zeros(3)

print("--- Hebbian Learning (AND Gate) ---")
for i in range(len(X)):
    # Weight update rule: w_new = w_old + x * y
    weights += X[i] * Y[i]
    print(f"Step {i+1} | Input: {X[i][:2]} Target: {Y[i]} | Weights: {weights}")

print("\nFinal Trained Weights:", weights)

# --- Visualization ---
# Plotting the data points
for i in range(len(X)):
    if Y[i] == 1:
        plt.scatter(X[i][0], X[i][1], color='green', marker='o', s=100, label='Class 1' if i == 3 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='red', marker='x', s=100, label='Class -1' if i == 0 else "")

# Plotting the decision boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
w1, w2, b = weights
if w2 != 0:
    x_val = np.array([-2, 2])
    y_val = -(w1 * x_val + b) / w2
    plt.plot(x_val, y_val, color='blue', linestyle='--', label='Decision Boundary')

# Formatting the plot
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Hebbian Learning - AND Gate')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
