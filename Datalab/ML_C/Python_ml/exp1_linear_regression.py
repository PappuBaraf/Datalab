# ML_Practical/exp1_linear_regression.py
# Experiment 1: To implementation of Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7, 8])

# Model Implementation
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Output
print("--- Linear Regression ---")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualization
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
