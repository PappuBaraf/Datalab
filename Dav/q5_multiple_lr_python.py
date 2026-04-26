# DAV_Practical/q5_multiple_lr_python.py
# Question 5: Implement and visualize multiple linear regression in Python. [cite: 2]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data: 2 Independent Variables (X1, X2), 1 Dependent Variable (y)
X = np.array([[1, 2], [2, 4], [3, 5], [4, 4], [5, 6], [6, 8], [7, 7]])
y = np.array([3, 5, 7, 6, 9, 11, 10])

# Model Implementation
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Output
print("--- Multiple Linear Regression (Python) ---")
print("Coefficients:", model.coef_)
print(f"Intercept: {model.intercept_:.4f}")
print(f"R^2 Score: {model.score(X, y):.4f}")

# Visualization (Actual vs Predicted)
plt.scatter(range(len(y)), y, color='blue', label='Actual', marker='o')
plt.plot(range(len(y)), y_pred, color='red', label='Predicted', marker='x')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()
