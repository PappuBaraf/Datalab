# DAV_Practical/q3_simple_lr_python.py
# Question 3: Implement and visualize simple linear regression in Python.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 3.5, 4, 5.5, 6, 8, 7.5, 9, 11, 12])

# Model Implementation
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Output Summary
print("--- Simple Linear Regression (Python) ---")
print(f"Coefficient (Slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualization (This is the plotting section!)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('X (Independent)')
plt.ylabel('y (Dependent)')
plt.legend()

# This command opens the window to display the plot
plt.show()
