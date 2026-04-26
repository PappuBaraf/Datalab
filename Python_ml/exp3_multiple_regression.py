# ML_Practical/exp3_multiple_regression.py
# Experiment 3: To implementation of Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generating a dataset with 3 features
X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

# Model Implementation
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Output
print("--- Multiple Linear Regression ---")
print("Coefficients:", model.coef_)
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared Score: {model.score(X, y):.4f}")
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.4f} | Actual: {y[i]:.4f}")

# --- Visualization ---
# For multiple regression, plotting Actual vs Predicted values is the standard approach
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.7, edgecolors='k', label='Data points')

# Plotting the ideal fit line (where Actual == Predicted)
min_val = min(np.min(y), np.min(y_pred))
max_val = max(np.max(y), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')

plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()